from __future__ import annotations

from pathlib import Path

from deepseek_distill.api import GenerationConfig
from scripts.collect_taco_breadth import build_campaign_manifest, build_parser


def test_breadth_cli_supports_low_memory_collect_only_mode() -> None:
    args = build_parser().parse_args(
        [
            "--tasks",
            "selected.jsonl",
            "--prior-tasks",
            "prior.jsonl",
            "--run-dir",
            "run",
            "--collect-only",
        ]
    )

    assert args.collect_only is True
    assert args.verifier_workers == 4


def test_breadth_manifest_freezes_one_attempt_without_embedding_tests(
    tmp_path: Path,
) -> None:
    tasks_path = tmp_path / "selected.jsonl"
    prior_path = tmp_path / "prior.jsonl"
    tasks_path.write_text("{}\n", encoding="utf-8")
    prior_path.write_text("{}\n{}\n", encoding="utf-8")
    tasks = [
        {"id": "taco_train_000010", "tests": [{"input": "SECRET_TEST"}]},
        {"id": "taco_train_000011", "tests": [{"input": "ANOTHER_SECRET_TEST"}]},
    ]

    manifest = build_campaign_manifest(
        all_tasks=tasks,
        run_tasks=tasks[:1],
        tasks_path=tasks_path,
        prior_tasks_path=prior_path,
        prior_task_count=2,
        config=GenerationConfig(max_tokens=4096),
        base_url="https://api.deepseek.com",
        phase_timeout=8,
    )

    assert manifest["dataset"]["excluded_prior_tasks"] == 2
    assert manifest["dataset"]["full_selected_tasks"] == 2
    assert manifest["dataset"]["run_tasks"] == 1
    assert manifest["sampling"] == {
        "max_attempts_per_task": 1,
        "blind": True,
        "verifier_feedback": False,
    }
    assert manifest["generation"]["max_tokens"] == 4096
    assert "SECRET_TEST" not in repr(manifest)
