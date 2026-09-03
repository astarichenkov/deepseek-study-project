"""Deterministic (zero-API-call) grading for the Day 3 built-in task.

Only used when the current task text equals the built-in demonstration task.
For any user-edited task we do NOT pretend to know the answer and return None.

Grading is conservative: we never claim a wrong answer because simple text
parsing could not confidently classify it.
"""
from __future__ import annotations

DEFAULT_REASONING_TASK = (
    "Анна, Борис и Виктор выступают с докладами в понедельник, вторник и среду — "
    "по одному человеку в день.\n\n"
    "Известно:\n\n"
    "1. Анна не выступает в понедельник.\n"
    "2. Борис выступает раньше Виктора.\n"
    "3. Виктор не выступает в среду.\n\n"
    "Определи, в какой день выступает каждый."
)

# person(lower) -> correct day(lower)
_REFERENCE = {
    "анна": "среда",
    "борис": "понедельник",
    "виктор": "вторник",
}

# morphological day forms -> canonical day
_DAY_ALIASES = [
    ("понедельник", "понедельник"),
    ("вторник", "вторник"),
    ("среда", "среда"),
    ("среду", "среда"),
]


def is_default_task(task: str) -> bool:
    """True when the given task text is (modulo whitespace) the built-in task."""
    return " ".join(task.split()) == " ".join(DEFAULT_REASONING_TASK.split())


def _day_positions(text: str):
    return [
        (pos, canonical)
        for (form, canonical) in _DAY_ALIASES
        for pos in _findall(text, form)
    ]


def _findall(text: str, needle: str):
    start = 0
    while True:
        i = text.find(needle, start)
        if i == -1:
            return
        yield i
        start = i + len(needle)


def _nearest_day(text: str, person: str) -> str | None:
    """Return the day that seems associated with the person (nearest day token
    within a window around a person mention), or None if unclear."""
    day_pos = sorted(_day_positions(text))
    if not day_pos:
        return None
    result = None
    best_dist = None
    for pos in _findall(text, person):
        for dpos, day in day_pos:
            dist = abs(pos - dpos)
            if dist < 80 and (best_dist is None or dist < best_dist):
                best_dist = dist
                result = day
    return result


import re


def _filter_negations(text: str) -> str:
    """Keep only sentences without a negation token ('не'). Restated conditions
    like 'Анна НЕ выступает в понедельник' are not final assignments and must
    not influence person->day detection."""
    parts = re.split(r"[.;!?]|\n", text)
    kept = [p for p in parts if not re.search(r"\bне\b", p)]
    return " ".join(kept)


def classify(text: str) -> str:
    """Return 'correct' | 'incorrect' | 'indeterminate' for the built-in task.

    * if any detected person->day (from a NON-negated statement) contradicts
      the reference -> 'incorrect'
    * if all three persons map to the correct day -> 'correct'
    * otherwise -> 'indeterminate' (we could not confidently parse it).
    """
    clean = _filter_negations(text.lower())
    detected = {p: _nearest_day(clean, p) for p in _REFERENCE}

    for person, expected in _REFERENCE.items():
        got = detected[person]
        if got is not None and got != expected:
            return "incorrect"
    if all(detected[p] == exp for p, exp in _REFERENCE.items()):
        return "correct"
    return "indeterminate"


def grade_task(task: str, answer: str) -> str | None:
    """Grade an answer. Returns None (no grading) for custom tasks."""
    if not is_default_task(task):
        return None
    return classify(answer)
