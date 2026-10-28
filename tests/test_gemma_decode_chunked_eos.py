from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import cts.backbone.gemma_adapter as gemma_adapter
from cts.backbone.gemma_adapter import GemmaCTSBackbone
from cts.perf.phase_profiler import CudaPhaseProfiler


class _TextConfig:
    hidden_size = 128
    num_hidden_layers = 4


class _Config:
    def get_text_config(self):
        return _TextConfig()


class _SequenceLM(nn.Module):
    def __init__(self, sequence: list[int]) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(128, 128)
        self.sequence = list(sequence)
        self.calls = 0

    def forward(self, *, inputs_embeds=None, input_ids=None, **kwargs):
        idx = min(self.calls, len(self.sequence) - 1)
        token_id = int(self.sequence[idx])
        self.calls += 1
        length = (
            int(inputs_embeds.shape[1])
            if inputs_embeds is not None
            else int(input_ids.shape[1])
        )
        hidden = torch.zeros(1, length, 128)
        hidden[:, -1, token_id] = 1.0
        return SimpleNamespace(
            last_hidden_state=hidden,
            past_key_values=(self.calls,),
        )


class _Tokenizer:
    bos_token_id = 2
    eos_token_id = 1
    chat_template = None

    def __call__(self, text, **kwargs):
        del text, kwargs
        return {
            "input_ids": torch.tensor([[2, 7]], dtype=torch.long),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
        }

    def decode(self, ids, *, skip_special_tokens=True):
        del skip_special_tokens
        return ",".join(str(int(x)) for x in ids)


def _backbone(sequence: list[int], monkeypatch) -> GemmaCTSBackbone:
    lm = _SequenceLM(sequence)
    cg = nn.Module()
    cg.config = _Config()
    cg.model = nn.Module()
    cg.model.language_model = lm
    cg.lm_head = nn.Identity()
    cg.generation_config = SimpleNamespace(eos_token_id=[1, 50, 106])
    monkeypatch.setattr(
        gemma_adapter,
        "raw_ple_for_mixed_public_forward",
        lambda *args, **kwargs: None,
    )
    return GemmaCTSBackbone(cg, _Tokenizer())


@pytest.mark.parametrize("eos_position", [1, 2, 3, 4, 5, 16, None])
@pytest.mark.parametrize("eos_id", [1, 50, 106])
def test_chunked_eos_check_preserves_greedy_prefix(
    monkeypatch, eos_position, eos_id
) -> None:
    max_tokens = 16
    sequence = [10 + (i % 20) for i in range(max_tokens)]
    if eos_position is not None:
        sequence[eos_position - 1] = eos_id

    baseline = _backbone(sequence, monkeypatch)
    chunked = _backbone(sequence, monkeypatch)
    z_star = torch.zeros(2, 128)

    baseline_text = baseline.decode_from_z_star(
        z_star,
        max_new_tokens=max_tokens,
        problem_text="test",
        eos_check_interval=1,
    )
    chunked_text = chunked.decode_from_z_star(
        z_star,
        max_new_tokens=max_tokens,
        problem_text="test",
        eos_check_interval=4,
    )

    assert chunked_text == baseline_text
    assert (
        chunked.last_decode_stats["generated_token_ids"]
        == baseline.last_decode_stats["generated_token_ids"]
    )
    assert (
        chunked.last_decode_stats["stop_reason"]
        == baseline.last_decode_stats["stop_reason"]
    )
    assert (
        chunked.last_decode_stats["decoder_forward_tokens"]
        - baseline.last_decode_stats["decoder_forward_tokens"]
        <= 3
    )
    assert chunked.last_decode_stats["generated_token_count"] == len(
        chunked.last_decode_stats["generated_token_ids"]
    )
    if eos_position is None:
        assert chunked.last_decode_stats["eos_emitted"] is False
        assert chunked.last_decode_stats["cap_reached"] is True
    else:
        assert chunked.last_decode_stats["eos_emitted"] is True
        assert chunked.last_decode_stats["cap_reached"] is False


@pytest.mark.parametrize("max_tokens", [1, 64])
def test_chunked_eos_check_respects_max_tokens(monkeypatch, max_tokens) -> None:
    sequence = [17] * max_tokens
    backbone = _backbone(sequence, monkeypatch)
    backbone.decode_from_z_star(
        torch.zeros(2, 128),
        max_new_tokens=max_tokens,
        problem_text="test",
        eos_check_interval=4,
    )
    assert len(backbone.last_decode_stats["generated_token_ids"]) == max_tokens
    assert backbone.last_decode_stats["generated_token_count"] == max_tokens
    assert backbone.last_decode_stats["stop_reason"] == "max_new_tokens"
    assert backbone.last_decode_stats["cap_reached"] is True
    assert backbone.last_decode_stats["eos_emitted"] is False
    assert backbone.last_decode_stats["cache_class"] == "tuple"
    assert backbone.last_decode_stats["prefix_tokens"] == 4
    assert backbone.last_decode_stats["problem_text_present"] is True


def test_decode_stats_record_chunk_overshoot_and_cache_reuse(monkeypatch) -> None:
    backbone = _backbone([10, 11, 106, 13, 14, 15], monkeypatch)
    backbone.decode_from_z_star(
        torch.zeros(2, 128),
        max_new_tokens=16,
        problem_text="test",
        eos_check_interval=4,
    )
    stats = backbone.last_decode_stats

    assert stats["schema_version"] == 2
    assert stats["generated_token_ids"] == [10, 11, 106]
    assert stats["generated_token_count"] == 3
    assert stats["decoder_forward_tokens"] == 5
    assert stats["discarded_after_eos"] == 2
    assert stats["host_eos_checks"] == 2
    assert stats["kv_cache_requested"] is True
    assert stats["kv_cache_prefill_present"] is True
    assert stats["kv_cache_type"] == "tuple"
    assert stats["kv_cache_reuse_calls"] == 4
    assert stats["observed_eos_token_id"] == 106
    assert stats["observed_eos_index"] == 2
    assert stats["eos_reached"] is True
    assert stats["cap_reached"] is False
    assert stats["soft_prompt_tokens"] == 2
    assert stats["text_prompt_tokens"] == 2
    assert stats["prefill_input_tokens"] == 4


def test_decode_stats_publish_only_after_call_finishes(monkeypatch) -> None:
    backbone = _backbone([17, 1], monkeypatch)
    previous = {"sentinel": True}
    backbone.last_decode_stats = previous
    original_decode = backbone.tokenizer.decode

    def observing_decode(ids, *, skip_special_tokens=True):
        assert backbone.last_decode_stats is previous
        return original_decode(ids, skip_special_tokens=skip_special_tokens)

    backbone.tokenizer.decode = observing_decode
    text = backbone.decode_from_z_star(
        torch.zeros(2, 128),
        max_new_tokens=4,
        problem_text="test",
        eos_check_interval=1,
    )

    assert text == "17,1"
    assert backbone.last_decode_stats is not previous
    assert backbone.last_decode_stats["generated_token_ids"] == [17, 1]
    assert backbone.last_decode_stats["decoded_char_count"] == len(text)


def test_decode_phase_profile_covers_all_four_phases(monkeypatch) -> None:
    backbone = _backbone([17, 1], monkeypatch)
    profiler = CudaPhaseProfiler(torch.device("cpu"))
    backbone.decode_from_z_star(
        torch.zeros(2, 128),
        max_new_tokens=4,
        problem_text="test",
        eos_check_interval=1,
        phase_profiler=profiler,
    )
    phases = profiler.finish()["phases"]

    assert set(phases) == {
        "decode.prepare",
        "decode.prefill",
        "decode.autoregressive",
        "decode.detokenize",
    }
