from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_taco_length_retry import _publish_retry_summary_once


def summary(*, mode: str, normalization_skipped: int, accepted: int = 2) -> dict:
    return {
        "schema_version": "coding.collection.taco.length_retry.summary.v2",
        "mode": mode,
        "collection": {"skipped": normalization_skipped},
        "normalization": {"skipped": normalization_skipped},
        "verification": {"skipped": normalization_skipped},
        "dataset": {"newly_accepted_tasks": accepted},
        "finish_reasons": {"length": 28, "stop": 14},
        "failure_counts": {"assertion_failure": 13},
        "outputs": {"newly_accepted_unique": mode},
    }


def test_retry_summary_rerun_ignores_only_per_invocation_counters(
    tmp_path: Path,
) -> None:
    path = tmp_path / "retry_summary.json"
    original = summary(mode="collect", normalization_skipped=0)
    rerun = summary(mode="aggregate_only", normalization_skipped=42)

    assert _publish_retry_summary_once(path, original) == "created"
    assert _publish_retry_summary_once(path, rerun) == "unchanged"
    assert json.loads(path.read_text(encoding="utf-8")) == original


def test_retry_summary_rerun_rejects_changed_dataset_outcome(tmp_path: Path) -> None:
    path = tmp_path / "retry_summary.json"
    _publish_retry_summary_once(
        path,
        summary(mode="collect", normalization_skipped=0),
    )

    with pytest.raises(FileExistsError, match="dataset"):
        _publish_retry_summary_once(
            path,
            summary(mode="aggregate_only", normalization_skipped=42, accepted=3),
        )
