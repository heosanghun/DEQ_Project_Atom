"""Gemma 4 E4B FP32 PLE-CPU local loader and forward equivalence runner.

Loads language model in FP32 with embed_tokens_per_layer (10,752 MiB) on CPU,
and transformer layers on RTX 4090 GPU (contiguous storage allocation).
Vision and audio towers are not loaded (미적재).
"""

from __future__ import annotations

import os
import sys
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoTokenizer, Gemma4ForConditionalGeneration
from safetensors import safe_open

DEFAULT_SNAPSHOT = "snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377"
DEFAULT_PLE_BIN = "D:/AI/cts/scratch/ple_fp32.bin"
PROMPT_JSONL = "data/stage2/math_train_prompts_5000.jsonl"


class CPUPLEWrapper(nn.Module):
    """Wrapper that holds embed_tokens_per_layer on CPU and transfers slices to GPU."""

    def __init__(self, ple_module: nn.Module, target_device: str | torch.device = "cuda:0"):
        super().__init__()
        object.__setattr__(self, "_ple_module", ple_module)
        object.__setattr__(self, "_target_device", torch.device(target_device))

    @property
    def weight(self) -> torch.Tensor:
        return self._ple_module.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids_cpu = input_ids.to("cpu")
        out_cpu = self._ple_module(input_ids_cpu)
        return out_cpu.to(self._target_device)

    def _apply(self, fn, recurse=True):
        return self

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None and device.type != "cpu":
            object.__setattr__(self, "_target_device", device)
        return self

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._ple_module, name)


def ensure_ple_binary(
    safetensors_path: str,
    out_bin_path: str = DEFAULT_PLE_BIN,
    ple_key: str = "model.language_model.embed_tokens_per_layer.weight",
) -> str:
    """Ensure PLE binary file exists on disk, extracting from safetensors if needed."""
    if os.path.exists(out_bin_path) and os.path.getsize(out_bin_path) == 262144 * 10752 * 4:
        return out_bin_path

    os.makedirs(os.path.dirname(os.path.abspath(out_bin_path)), exist_ok=True)
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        st_slice = f.get_slice(ple_key)
        fp = np.memmap(out_bin_path, dtype="float32", mode="w+", shape=(262144, 10752))
        chunk_rows = 16384
        for start in range(0, 262144, chunk_rows):
            end = min(start + chunk_rows, 262144)
            chunk = st_slice[start:end].to(dtype=torch.float32).numpy()
            fp[start:end] = chunk
            del chunk
        fp.flush()
        del fp
    return out_bin_path


def load_gemma4_fp32_ple_cpu(
    snapshot_dir: str = DEFAULT_SNAPSHOT,
    ple_bin_path: str = DEFAULT_PLE_BIN,
    target_device: str = "cuda:0",
    verbose: bool = True,
) -> Tuple[Gemma4ForConditionalGeneration, AutoTokenizer]:
    """Load Gemma 4 LM in FP32 with PLE on CPU and remaining layers in contiguous GPU storage."""
    safetensors_path = os.path.join(snapshot_dir, "model.safetensors")
    ensure_ple_binary(safetensors_path, ple_bin_path)

    cfg = AutoConfig.from_pretrained(snapshot_dir)
    with init_empty_weights():
        model = Gemma4ForConditionalGeneration(cfg)

    # Multimodal 미적재 (vision/audio towers set to None)
    model.model.vision_tower = None
    model.model.audio_tower = None
    model.model.embed_vision = None
    model.model.embed_audio = None

    model_state = model.state_dict()
    model_keys = set(model_state.keys())
    ple_key = "model.language_model.embed_tokens_per_layer.weight"

    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        st_keys = set(f.keys())
        gpu_keys = []
        total_gpu_numel = 0

        for k in sorted(list(model_keys)):
            if k == "lm_head.weight" or k == ple_key:
                continue
            if k in st_keys:
                shape = f.get_slice(k).get_shape()
                numel = 1
                for dim in shape:
                    numel *= dim
                gpu_keys.append((k, shape, numel))
                total_gpu_numel += numel

        if verbose:
            print(f"[LOADER] GPU keys count: {len(gpu_keys)}, total numel: {total_gpu_numel} ({total_gpu_numel*4/(1024**2):.2f} MiB)")

        # 1. Load PLE-CPU from mmap binary
        mmap_arr = np.memmap(ple_bin_path, dtype="float32", mode="r", shape=(262144, 10752))
        ple_tensor = torch.from_numpy(mmap_arr)
        model.model.language_model.embed_tokens_per_layer.weight = nn.Parameter(ple_tensor, requires_grad=False)

        # 2. Allocate contiguous GPU storage
        gpu_storage = torch.empty((total_gpu_numel,), dtype=torch.float32, device=target_device)
        if verbose:
            print(f"[LOADER] Contiguous GPU storage allocated: {torch.cuda.memory_allocated()/(1024**2):.2f} MiB")

        # 3. Copy weights directly into storage slices
        t0 = time.time()
        offset = 0
        for idx, (key, shape, numel) in enumerate(gpu_keys):
            slice_gpu = gpu_storage[offset : offset + numel].view(shape)
            tensor_cpu = f.get_tensor(key).to(dtype=torch.float32)
            slice_gpu.copy_(tensor_cpu, non_blocking=True)

            parts = key.split(".")
            mod = model
            for p in parts[:-1]:
                mod = getattr(mod, p)
            param_name = parts[-1]

            if param_name in mod._buffers:
                mod._buffers[param_name] = slice_gpu
            else:
                mod._parameters[param_name] = nn.Parameter(slice_gpu, requires_grad=False)

            offset += numel

        torch.cuda.synchronize()
        if verbose:
            print(f"[LOADER] GPU weights loaded in {time.time()-t0:.2f}s")

    # Tie lm_head
    model.lm_head.weight = model.model.language_model.embed_tokens.weight

    # Wrap PLE module
    model.model.language_model.embed_tokens_per_layer = CPUPLEWrapper(
        model.model.language_model.embed_tokens_per_layer,
        target_device=target_device,
    )

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(snapshot_dir)
    return model, tokenizer


def load_gemma4_fp32_all_cpu(
    snapshot_dir: str = DEFAULT_SNAPSHOT,
    ple_bin_path: str = DEFAULT_PLE_BIN,
    verbose: bool = True,
) -> Tuple[Gemma4ForConditionalGeneration, AutoTokenizer]:
    """Load Gemma 4 LM in FP32 with all parameters in CPU memory."""
    safetensors_path = os.path.join(snapshot_dir, "model.safetensors")
    ensure_ple_binary(safetensors_path, ple_bin_path)

    cfg = AutoConfig.from_pretrained(snapshot_dir)
    with init_empty_weights():
        model = Gemma4ForConditionalGeneration(cfg)

    model.model.vision_tower = None
    model.model.audio_tower = None
    model.model.embed_vision = None
    model.model.embed_audio = None

    model_state = model.state_dict()
    model_keys = set(model_state.keys())
    ple_key = "model.language_model.embed_tokens_per_layer.weight"

    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        st_keys = set(f.keys())
        cpu_keys = []
        total_cpu_numel = 0

        for k in sorted(list(model_keys)):
            if k == "lm_head.weight" or k == ple_key:
                continue
            if k in st_keys:
                shape = f.get_slice(k).get_shape()
                numel = 1
                for dim in shape:
                    numel *= dim
                cpu_keys.append((k, shape, numel))
                total_cpu_numel += numel

        if verbose:
            print(f"[CPU LOADER] CPU keys count: {len(cpu_keys)}, total numel: {total_cpu_numel} ({total_cpu_numel*4/(1024**2):.2f} MiB)")

        mmap_arr = np.memmap(ple_bin_path, dtype="float32", mode="r", shape=(262144, 10752))
        ple_tensor = torch.from_numpy(mmap_arr)
        model.model.language_model.embed_tokens_per_layer.weight = nn.Parameter(ple_tensor, requires_grad=False)

        cpu_storage = torch.empty((total_cpu_numel,), dtype=torch.float32, device="cpu")

        t0 = time.time()
        offset = 0
        for idx, (key, shape, numel) in enumerate(cpu_keys):
            slice_cpu = cpu_storage[offset : offset + numel].view(shape)
            tensor = f.get_tensor(key).to(dtype=torch.float32)
            slice_cpu.copy_(tensor)

            parts = key.split(".")
            mod = model
            for p in parts[:-1]:
                mod = getattr(mod, p)
            param_name = parts[-1]

            if param_name in mod._buffers:
                mod._buffers[param_name] = slice_cpu
            else:
                mod._parameters[param_name] = nn.Parameter(slice_cpu, requires_grad=False)

            offset += numel

        if verbose:
            print(f"[CPU LOADER] CPU weights loaded in {time.time()-t0:.2f}s")

    model.lm_head.weight = model.model.language_model.embed_tokens.weight

    model.model.language_model.embed_tokens_per_layer = CPUPLEWrapper(
        model.model.language_model.embed_tokens_per_layer,
        target_device="cpu",
    )

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(snapshot_dir)
    return model, tokenizer


def run_phase_cpu(prompt_text: str, out_file: str = "D:/AI/cts/scratch/cpu_ref_outputs.pt") -> Dict[str, Any]:
    """Phase 1: Pure CPU FP32 baseline execution in standalone process."""
    print("=== PHASE 1: [All CPU FP32] ===")
    t0_load = time.time()
    model_cpu, tokenizer = load_gemma4_fp32_all_cpu(verbose=True)
    t_load = time.time() - t0_load

    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_ids = inputs.input_ids
    seq_len = int(input_ids.shape[1])
    print(f"[CPU] Tokenized sequence length: {seq_len}")

    t0_fwd = time.time()
    with torch.no_grad():
        out = model_cpu(input_ids=input_ids, output_hidden_states=True)
        logits = out.logits.detach().cpu()
        hidden = out.hidden_states[-1].detach().cpu()
    t_fwd = time.time() - t0_fwd
    print(f"[CPU] Forward completed in {t_fwd:.2f}s")

    torch.save({"logits": logits, "hidden": hidden, "seq_len": seq_len, "t_load_s": t_load, "t_fwd_s": t_fwd}, out_file)
    print(f"[CPU] Reference outputs saved to {out_file}")
    return {"seq_len": seq_len, "t_load_s": t_load, "t_fwd_s": t_fwd}


def run_phase_gpu(prompt_text: str, ref_file: str = "D:/AI/cts/scratch/cpu_ref_outputs.pt") -> Dict[str, Any]:
    """Phase 2: PLE-CPU GPU FP32 execution and delta calculation in standalone process."""
    print("=== PHASE 2: [PLE-CPU + GPU FP32] ===")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    tf32_matmul = bool(torch.backends.cuda.matmul.allow_tf32)
    tf32_cudnn = bool(torch.backends.cudnn.allow_tf32)

    torch.cuda.reset_peak_memory_stats()
    t0_load = time.time()
    model_gpu, tokenizer = load_gemma4_fp32_ple_cpu(verbose=True)
    t_load = time.time() - t0_load

    vram_after_load_mib = float(torch.cuda.memory_allocated() / (1024**2))
    max_vram_load_mib = float(torch.cuda.max_memory_allocated() / (1024**2))

    inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda:0")

    t0_fwd = time.time()
    with torch.no_grad():
        out = model_gpu(input_ids=inputs.input_ids, output_hidden_states=True)
        logits_gpu = out.logits.detach().cpu()
        hidden_gpu = out.hidden_states[-1].detach().cpu()
    t_fwd = time.time() - t0_fwd

    vram_after_fwd_mib = float(torch.cuda.memory_allocated() / (1024**2))
    max_vram_peak_mib = float(torch.cuda.max_memory_allocated() / (1024**2))

    # Load CPU reference
    ref = torch.load(ref_file, map_location="cpu", weights_only=True)
    logits_cpu = ref["logits"]
    hidden_cpu = ref["hidden"]

    delta_logits = (logits_cpu - logits_gpu).abs()
    max_delta_logits = float(delta_logits.max().item())
    mean_delta_logits = float(delta_logits.mean().item())

    delta_hidden = (hidden_cpu - hidden_gpu).abs()
    max_delta_hidden = float(delta_hidden.max().item())
    mean_delta_hidden = float(delta_hidden.mean().item())

    cos_sim = float(
        torch.nn.functional.cosine_similarity(
            hidden_cpu.view(-1, hidden_cpu.shape[-1]),
            hidden_gpu.view(-1, hidden_gpu.shape[-1]),
            dim=-1,
        ).mean().item()
    )

    results = {
        "tf32_matmul": tf32_matmul,
        "tf32_cudnn": tf32_cudnn,
        "seq_len": ref["seq_len"],
        "t_cpu_load_s": ref["t_load_s"],
        "t_cpu_fwd_s": ref["t_fwd_s"],
        "t_gpu_load_s": t_load,
        "t_gpu_fwd_s": t_fwd,
        "vram_after_load_mib": vram_after_load_mib,
        "max_vram_load_mib": max_vram_load_mib,
        "vram_after_fwd_mib": vram_after_fwd_mib,
        "max_vram_peak_mib": max_vram_peak_mib,
        "max_delta_logits": max_delta_logits,
        "mean_delta_logits": mean_delta_logits,
        "max_delta_hidden": max_delta_hidden,
        "mean_delta_hidden": mean_delta_hidden,
        "cos_sim_hidden": cos_sim,
    }

    print("\n=== FINAL EQUIVALENCE BENCHMARK RESULTS ===")
    for k, v in results.items():
        print(f"  {k}: {v}")

    return results


if __name__ == "__main__":
    import argparse
    import subprocess

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["cpu", "gpu", "all"], default="all")
    args = parser.parse_args()

    with open(PROMPT_JSONL, "r", encoding="utf-8") as f:
        first_line = json.loads(f.readline())
        prompt = first_line["prompt"]

    ref_path = "D:/AI/cts/scratch/cpu_ref_outputs.pt"

    if args.phase == "cpu":
        run_phase_cpu(prompt, ref_path)
    elif args.phase == "gpu":
        res = run_phase_gpu(prompt, ref_path)
        with open("D:/AI/cts/scratch/benchmark_results.json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
    else:
        # Run CPU phase in separate subprocess to allow OS to reclaim 17.4 GB commit
        print("[ORCHESTRATOR] Spawning CPU phase subprocess...")
        p_cpu = subprocess.run([sys.executable, __file__, "--phase", "cpu"], check=True)
        print("[ORCHESTRATOR] CPU phase complete. Spawning GPU phase subprocess...")
        p_gpu = subprocess.run([sys.executable, __file__, "--phase", "gpu"], check=True)
        print("[ORCHESTRATOR] All phases complete.")
