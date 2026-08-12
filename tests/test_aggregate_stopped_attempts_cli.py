from __future__ import annotations

from pathlib import Path


def test_partial_aggregator_never_normalizes_verifies_or_calls_api(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from scripts import aggregate_stopped_attempts as cli

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    selected = tmp_path / "selected.jsonl"
    selected.write_text("{}\n", encoding="utf-8")
    for name in ("raw_attempts.jsonl", "normalized_attempts.jsonl", "verifier_attempts.jsonl"):
        (run_dir / name).write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        cli,
        "aggregate_attempt_campaign",
        lambda **kwargs: {"counts": {"accepted_unique": 7}},
    )
    summary = cli.aggregate_stopped_attempts(
        run_dir=run_dir,
        selected_tasks_path=selected,
        target=10,
        max_attempts_per_task=3,
    )

    assert summary["api_requests"] == 0
    assert summary["normalization_started"] is False
    assert summary["verification_started"] is False
    assert summary["aggregate"]["counts"]["accepted_unique"] == 7


def test_partial_aggregator_requires_all_append_only_inputs(tmp_path: Path) -> None:
    from scripts.aggregate_stopped_attempts import aggregate_stopped_attempts

    selected = tmp_path / "selected.jsonl"
    selected.write_text("{}\n", encoding="utf-8")
    try:
        aggregate_stopped_attempts(
            run_dir=tmp_path,
            selected_tasks_path=selected,
            target=1,
            max_attempts_per_task=3,
        )
    except FileNotFoundError as error:
        assert "raw_attempts.jsonl" in str(error)
    else:
        raise AssertionError("missing append-only inputs must fail")
