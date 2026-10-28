from __future__ import annotations

import ast
from pathlib import Path


def _load_build_prompt():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_cts_eval_full.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "_build_prompt"
    )
    module = ast.Module(body=[node], type_ignores=[])
    namespace: dict[str, object] = {}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["_build_prompt"]


_build_prompt = _load_build_prompt()


def test_greedy_math_prompt_uses_validated_concise_box_contract():
    prompt = _build_prompt("What is 2+3?", "math500", native_think=False)

    assert "at most four short sentences" in prompt
    assert "exactly one line \\boxed{FINAL_ANSWER}" in prompt
    assert prompt.endswith("What is 2+3?")
    assert "<|turn>" not in prompt
    assert "<start_of_turn>" not in prompt


def test_native_math_prompt_is_user_content_not_handwritten_chat_markup():
    prompt = _build_prompt("What is 2+3?", "math500", native_think=True)

    assert "\\boxed{FINAL_ANSWER}" in prompt
    assert "<|turn>" not in prompt
    assert "<start_of_turn>" not in prompt
