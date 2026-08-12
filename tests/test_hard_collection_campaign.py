from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_hard_collection_campaign import (
    build_collect_command,
    build_import_command,
    load_campaign_config,
)


def _config() -> dict:
    return {
        "schema_version": "coding.collection.hard_overnight.v1",
        "campaign_id": "unit-hard",
        "run_root": "data/unit-hard",
        "cache_dir": "cache",
        "minimum_free_gib": 1,
        "stop_free_gib": 0.25,
        "difficulty_profile": "hard-v1",
        "generation": {
            "model": "deepseek-v4-pro",
            "temperature": 0.2,
            "top_p": 1.0,
            "top_logprobs": 20,
            "max_tokens": 8192,
            "timeout": 180,
            "max_retries": 2,
        },
        "sampling": {"max_attempts_per_task": 3},
        "verification": {
            "phase_timeout": 12,
            "max_output_characters": 131072,
        },
        "budgets": {
            "max_total_api_workers": 32,
            "max_total_verifier_workers": 16,
            "max_total_requests_per_minute": 120,
        },
        "lanes": [
            {
                "name": "cf-hard",
                "source": "open-r1-codeforces",
                "limit": 10,
                "seed": 20260804,
                "exclude_tasks": ["data/prior-cf-tasks.jsonl"],
                "api_workers": 8,
                "verifier_workers": 4,
                "requests_per_minute": 30,
            }
        ],
    }


def test_hard_campaign_commands_freeze_profile_and_verified_generation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(_config()), encoding="utf-8")
    config = load_campaign_config(path, repo_root=tmp_path)
    lane = config.lanes[0]

    import_command = build_import_command(config, lane, python="python.exe")
    collect_command = build_collect_command(config, lane, python="python.exe")

    assert import_command[:2] == ["python.exe", str(tmp_path / "scripts/import_multisource.py")]
    assert import_command[import_command.index("--difficulty-profile") + 1] == "hard-v1"
    assert import_command[import_command.index("--limit") + 1] == "10"
    exclusion_index = import_command.index("--exclude-tasks")
    assert import_command[exclusion_index + 1] == str(
        tmp_path / "data/prior-cf-tasks.jsonl"
    )
    assert collect_command[collect_command.index("--max-tokens") + 1] == "8192"
    assert collect_command[collect_command.index("--top-logprobs") + 1] == "20"
    assert collect_command[collect_command.index("--workers") + 1] == "8"
    assert collect_command[collect_command.index("--max-attempts-per-task") + 1] == "3"
    assert collect_command[collect_command.index("--verifier-workers") + 1] == "4"
    assert "--streaming-pipeline" in collect_command
    assert not any("API_KEY" in item for item in import_command + collect_command)


def test_hard_campaign_threads_actual_only_trace_profile_without_top_k(
    tmp_path: Path,
) -> None:
    raw = _config()
    raw["generation"]["trace_profile"] = "actual_only"
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    config = load_campaign_config(path, repo_root=tmp_path)
    collect_command = build_collect_command(
        config,
        config.lanes[0],
        python="python.exe",
    )

    assert collect_command[collect_command.index("--trace-profile") + 1] == (
        "actual_only"
    )
    assert "--top-logprobs" not in collect_command


def test_hard_campaign_rejects_unknown_trace_profile(tmp_path: Path) -> None:
    raw = _config()
    raw["generation"]["trace_profile"] = "unknown"
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="trace_profile"):
        load_campaign_config(path, repo_root=tmp_path)


def test_hard_campaign_can_reuse_supervisor_without_difficulty_filter(
    tmp_path: Path,
) -> None:
    raw = _config()
    raw["difficulty_profile"] = None
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    config = load_campaign_config(path, repo_root=tmp_path)
    import_command = build_import_command(
        config,
        config.lanes[0],
        python="python.exe",
    )

    assert config.difficulty_profile is None
    assert "--difficulty-profile" not in import_command


def test_hard_campaign_rejects_worker_budget_oversubscription(tmp_path: Path) -> None:
    raw = _config()
    raw["lanes"].append(
        {
            "name": "taco-hard",
            "source": "taco-multishard",
            "limit": 10,
            "seed": 20260805,
            "api_workers": 25,
            "verifier_workers": 13,
            "requests_per_minute": 100,
        }
    )
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="API worker budget"):
        load_campaign_config(path, repo_root=tmp_path)


def test_hard_campaign_rejects_non_hard_or_duplicate_lanes(tmp_path: Path) -> None:
    raw = _config()
    raw["difficulty_profile"] = "hard-v2"
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="difficulty_profile"):
        load_campaign_config(path, repo_root=tmp_path)

    raw = _config()
    raw["lanes"].append(dict(raw["lanes"][0]))
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate lane"):
        load_campaign_config(path, repo_root=tmp_path)


def test_hard_campaign_rejects_invalid_exclusion_paths(tmp_path: Path) -> None:
    raw = _config()
    raw["lanes"][0]["exclude_tasks"] = "data/prior.jsonl"
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="exclude_tasks"):
        load_campaign_config(path, repo_root=tmp_path)
