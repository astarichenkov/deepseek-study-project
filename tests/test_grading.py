"""Tests for the deterministic (zero-API-call) Day 3 grading."""
import pytest

from app.services.grading import (
    DEFAULT_REASONING_TASK,
    classify,
    grade_task,
    is_default_task,
)

CORRECT_PLAIN = "Борис — понедельник. Виктор — вторник. Анна — среда."
CORRECT_WORDED = "Ответ: Борис выступает в понедельник, Виктор во вторник, а Анна в среду."
INCORRECT = "Борис — среда. Виктор — понедельник. Анна — вторник."


def test_is_default_task_exact_and_whitespace_insensitive():
    assert is_default_task(DEFAULT_REASONING_TASK)
    assert is_default_task("  " + DEFAULT_REASONING_TASK + "  ")
    assert not is_default_task("Какая-то другая задача о погоде.")


def test_classify_correct_reference_wording():
    assert classify(CORRECT_PLAIN) == "correct"


def test_classify_correct_despite_harmless_wording():
    assert classify(CORRECT_WORDED) == "correct"


def test_classify_incorrect_assignment():
    assert classify(INCORRECT) == "incorrect"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Не знаю, но скорее всего где-то в будние дни.",
        "Анна выступает после Бориса.",
    ],
)
def test_classify_conservative_on_unparseable(text):
    # We never claim 'correct' unless confidently parsed; unclear text is not
    # marked wrong merely because we could not parse it.
    assert classify(text) != "correct"


def test_grade_default_task_returns_status():
    assert grade_task(DEFAULT_REASONING_TASK, CORRECT_PLAIN) == "correct"
    assert grade_task(DEFAULT_REASONING_TASK, INCORRECT) == "incorrect"
    assert grade_task(DEFAULT_REASONING_TASK, "Странный ответ") == "indeterminate"


def test_grade_custom_task_returns_none():
    assert grade_task("Своя задача про города.", "неважно") is None


def test_grade_makes_no_provider_calls():
    # grading is pure text classification - nothing is called
    assert grade_task(DEFAULT_REASONING_TASK, CORRECT_PLAIN) is not None


RESTATED_CORRECT = (
    "Условия: Анна не выступает в понедельник; Борис раньше Виктора; "
    "Виктор не выступает в среду. Рассуждаю: Анна не в понедельник, "
    "значит ... Итог: Борис выступает в понедельник, Виктор во вторник, "
    "Анна в среду."
)


def test_classify_correct_when_conditions_restated():
    # Negated restatements must not be read as assignments.
    assert classify(RESTATED_CORRECT) == "correct"
