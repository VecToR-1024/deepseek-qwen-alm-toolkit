from __future__ import annotations

import pytest

from deepseek_distill.hard_tasks import (
    HARD_DIFFICULTY_PROFILE,
    is_hard_task,
)


@pytest.mark.parametrize(
    ("source", "metadata", "expected"),
    [
        ("open-r1-codeforces", {"rating": 1400}, True),
        ("open-r1-codeforces", {"rating": 1300}, False),
        ("open-r1-codeforces", {"rating": None}, False),
        ("taco-multishard", {"difficulty": "MEDIUM_HARD"}, True),
        ("taco-multishard", {"difficulty": "hard"}, True),
        ("taco-multishard", {"difficulty": "VERY_HARD"}, True),
        ("taco-multishard", {"difficulty": "MEDIUM"}, False),
        ("apps", {"difficulty": "competition"}, True),
        ("apps", {"difficulty": "interview"}, False),
        ("code-contests", {"codeforces": {"rating": 1500}, "difficulty": 8}, True),
        ("code-contests", {"codeforces": {"rating": 1200}, "difficulty": 12}, False),
        ("code-contests", {"codeforces": {"rating": 0}, "difficulty": 3}, True),
        ("code-contests", {"codeforces": {"rating": 0}, "difficulty": 5}, True),
        ("code-contests", {"codeforces": {"rating": 0}, "difficulty": 9}, True),
        ("code-contests", {"codeforces": {"rating": 0}, "difficulty": 8}, False),
    ],
)
def test_hard_v1_profile_has_explicit_source_specific_thresholds(
    source: str,
    metadata: dict,
    expected: bool,
) -> None:
    task = {"metadata": metadata}

    assert is_hard_task(source, task, profile=HARD_DIFFICULTY_PROFILE) is expected


def test_hard_profile_rejects_unsupported_sources() -> None:
    with pytest.raises(ValueError, match="does not support"):
        is_hard_task("odex", {"metadata": {}}, profile=HARD_DIFFICULTY_PROFILE)


def test_hard_profile_rejects_unknown_profile() -> None:
    with pytest.raises(ValueError, match="unknown difficulty profile"):
        is_hard_task("apps", {"metadata": {}}, profile="hard-v999")
