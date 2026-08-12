"""Auditable source-specific difficulty policy for hard-data collection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


HARD_DIFFICULTY_PROFILE = "hard-v1"
HARD_PROFILE_SOURCES = frozenset(
    {
        "apps",
        "code-contests",
        "open-r1-codeforces",
        "taco-multishard",
    }
)


def is_hard_task(
    source: str,
    task: Mapping[str, Any],
    *,
    profile: str = HARD_DIFFICULTY_PROFILE,
) -> bool:
    """Return whether a normalized task satisfies the frozen hard-v1 policy."""

    if profile != HARD_DIFFICULTY_PROFILE:
        raise ValueError(f"unknown difficulty profile {profile!r}")
    if source not in HARD_PROFILE_SOURCES:
        raise ValueError(f"difficulty profile {profile!r} does not support {source!r}")
    metadata = task.get("metadata")
    if not isinstance(metadata, Mapping):
        return False

    if source == "open-r1-codeforces":
        return _number_at_least(metadata.get("rating"), 1400)
    if source == "taco-multishard":
        return _difficulty_name(metadata.get("difficulty")) in {
            "MEDIUM_HARD",
            "HARD",
            "VERY_HARD",
        }
    if source == "apps":
        return _difficulty_name(metadata.get("difficulty")) == "COMPETITION"

    codeforces = metadata.get("codeforces")
    rating = codeforces.get("rating") if isinstance(codeforces, Mapping) else None
    if _positive_number(rating):
        return _number_at_least(rating, 1400)
    difficulty = metadata.get("difficulty")
    return (
        isinstance(difficulty, int)
        and not isinstance(difficulty, bool)
        and (difficulty in {3, 4, 5} or difficulty >= 9)
    )


def hard_profile_metadata() -> dict[str, Any]:
    """Return a JSON-serializable description suitable for run manifests."""

    return {
        "id": HARD_DIFFICULTY_PROFILE,
        "rules": {
            "open-r1-codeforces": {"rating_min": 1400, "missing_rating": "reject"},
            "taco-multishard": {
                "difficulty_in": ["MEDIUM_HARD", "HARD", "VERY_HARD"]
            },
            "apps": {"difficulty_in": ["COMPETITION"]},
            "code-contests": {
                "rated_codeforces": {"rating_min": 1400},
                "unrated_fallback_difficulty_values": [3, 4, 5, "C_OR_LATER"],
                "unrated_fallback_numeric_min_for_problem_letter": 9,
            },
        },
    }


def _difficulty_name(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().upper().replace("-", "_").replace(" ", "_")


def _positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def _number_at_least(value: Any, threshold: int) -> bool:
    return _positive_number(value) and value >= threshold
