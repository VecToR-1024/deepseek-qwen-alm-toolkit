from __future__ import annotations

from pathlib import Path

from deepseek_distill.code_verifier import VerificationSummary
from deepseek_distill.normalize import AppendNormalizeSummary


def test_offline_finalizer_drains_and_aggregates_without_api_configuration(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from scripts import finalize_stopped_attempts as finalizer

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    selected_tasks = tmp_path / "selected.jsonl"
    selected_tasks.write_text("{}\n", encoding="utf-8")
    (run_dir / "raw_attempts.jsonl").write_text("{}\n", encoding="utf-8")
    calls: list[tuple[str, object]] = []

    def fake_normalize(input_path, output_path, *, error_output_path):
        calls.append(("normalize", (input_path, output_path, error_output_path)))
        return AppendNormalizeSummary(1, 0, 1, 0, 0, 0)

    def fake_project(**kwargs):
        calls.append(("project", kwargs))
        return 0

    def fake_verify(**kwargs):
        calls.append(("verify", kwargs))
        return VerificationSummary(1, 0, 1, 0, {})

    def fake_aggregate(**kwargs):
        calls.append(("aggregate", kwargs))
        return {"counts": {"accepted_unique": 1}}

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(finalizer, "normalize_jsonl_append", fake_normalize)
    monkeypatch.setattr(
        finalizer,
        "append_normalization_failures_to_verifier",
        fake_project,
    )
    monkeypatch.setattr(finalizer, "verify_jsonl", fake_verify)
    monkeypatch.setattr(finalizer, "aggregate_attempt_campaign", fake_aggregate)

    summary = finalizer.finalize_stopped_attempts(
        run_dir=run_dir,
        selected_tasks_path=selected_tasks,
        target=1,
        max_attempts_per_task=3,
        verifier_workers=4,
        phase_timeout_seconds=12,
        max_output_characters=131_072,
    )

    assert [name for name, _ in calls] == [
        "normalize",
        "project",
        "verify",
        "aggregate",
    ]
    assert calls[2][1]["max_workers"] == 4
    assert summary["api_requests"] == 0
    assert summary["aggregate"]["counts"]["accepted_unique"] == 1


def test_offline_finalizer_requires_existing_raw_and_selected_files(
    tmp_path: Path,
) -> None:
    from scripts.finalize_stopped_attempts import finalize_stopped_attempts

    try:
        finalize_stopped_attempts(
            run_dir=tmp_path / "missing-run",
            selected_tasks_path=tmp_path / "missing-selected.jsonl",
            target=1,
            max_attempts_per_task=3,
            verifier_workers=1,
            phase_timeout_seconds=12,
            max_output_characters=1024,
        )
    except FileNotFoundError as error:
        assert "selected task" in str(error)
    else:
        raise AssertionError("missing inputs must fail before processing")
