from cts.train.openmath_text import (
    prompt_text_from_openmath_row,
    solution_text_from_openmath_row,
    target_text_from_openmath_row,
)


def test_openmath_v1_question_field():
    """OpenMathInstruct-1 schema uses `question`."""
    row = {"question": " 2+2? ", "expected_answer": "4"}
    assert prompt_text_from_openmath_row(row) == "2+2?"


def test_openmath_v2_problem_field_is_canonical():
    """OpenMathInstruct-2 (paper §6.1 canonical) uses `problem`."""
    row = {"problem": "x"}
    assert prompt_text_from_openmath_row(row) == "x"


def test_openmath_v2_takes_priority_over_v1_when_both_present():
    """If both schemas are present in a row (defensive), prefer the v2
    `problem` key since v2 is the paper-canonical corpus."""
    row = {"problem": "v2-canonical", "question": "v1-legacy"}
    assert prompt_text_from_openmath_row(row) == "v2-canonical"


def test_openmath_falls_back_to_messages_list():
    """Some HF revisions store conversational turns under `messages`."""
    row = {"messages": [{"content": "hi"}, {"content": "world"}]}
    out = prompt_text_from_openmath_row(row)
    assert "hi" in out and "world" in out


def test_openmath_handles_non_dict_row():
    assert prompt_text_from_openmath_row("plain string") == "plain string"


def test_openmath_skips_empty_strings():
    row = {"problem": "", "question": "real prompt"}
    assert prompt_text_from_openmath_row(row) == "real prompt"


def test_openmath_target_prefers_gold_answer_for_terminal_wproj():
    row = {
        "generated_solution": "work, then \\boxed{4}",
        "expected_answer": "4",
    }
    assert target_text_from_openmath_row(row) == "\\boxed{4}"


def test_openmath_target_wraps_short_answer_when_solution_missing():
    assert target_text_from_openmath_row({"expected_answer": "29"}) == "\\boxed{29}"


def test_openmath_target_does_not_double_box_gold_answer():
    assert target_text_from_openmath_row({"answer": "\\boxed{1/2}"}) == "\\boxed{1/2}"


def test_openmath_target_uses_derivation_only_without_gold_answer():
    assert target_text_from_openmath_row({"generated_solution": "work"}) == "work"


def test_openmath_solution_preserves_derivation_separately_from_final_answer():
    row = {
        "generated_solution": "derive step by step; therefore 4",
        "expected_answer": "4",
    }
    assert solution_text_from_openmath_row(row) == row["generated_solution"]
    assert target_text_from_openmath_row(row) == "\\boxed{4}"
