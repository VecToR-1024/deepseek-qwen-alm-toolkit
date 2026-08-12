from __future__ import annotations

import json
import re
from pathlib import Path

from scripts.run_hard_collection_campaign import (
    build_collect_command,
    load_campaign_config,
)


ROOT = Path(__file__).resolve().parents[1]


def test_actual_only_example_uses_the_48_worker_topology() -> None:
    path = ROOT / "configs" / "collection.actual-only.48workers.example.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    config = load_campaign_config(path, repo_root=ROOT)

    assert sum(lane.api_workers for lane in config.lanes) == 32
    assert sum(lane.verifier_workers for lane in config.lanes) == 16
    assert raw["generation"]["trace_profile"] == "actual_only"
    for lane in config.lanes:
        command = build_collect_command(config, lane, python="python")
        assert "--streaming-pipeline" in command
        assert "--top-logprobs" not in command


def test_training_arms_share_one_entrypoint_and_only_change_objective_weight() -> None:
    sft = json.loads((ROOT / "configs" / "training.sft-only.example.json").read_text())
    alm = json.loads((ROOT / "configs" / "training.sft-alm.example.json").read_text())

    assert sft["entrypoint"] == alm["entrypoint"] == "examples/train_offline_alm.py"
    assert sft["environment"] | {"ALPHA_ALM": "10.0"} == alm["environment"]
    assert sft["environment"]["ALPHA_ALM"] == "0.0"
    assert sft["training_mode"] == alm["training_mode"] == "bf16_lora"


def test_qwen3_full_finetune_template_disables_thinking_and_lora() -> None:
    config = json.loads(
        (ROOT / "configs" / "training.qwen3-0.6b-full.example.json").read_text()
    )
    environment = config["environment"]

    assert config["entrypoint"] == "examples/train_offline_alm.py"
    assert config["validation_status"] == "data_contract_only"
    assert config["training_mode"] == "bf16_full"
    assert environment["STUDENT_MODEL"] == "Qwen/Qwen3-0.6B"
    assert environment["USE_LORA"] == "0"
    assert json.loads(environment["CHAT_TEMPLATE_KWARGS"]) == {
        "enable_thinking": False
    }

    launcher = (
        ROOT / "examples" / "training" / "launch_qwen3_0_6b_full_pair.sh"
    ).read_text()
    assert "dataset_sha256=" in launcher
    assert "dataset_records=" in launcher
    assert "toolkit_commit=" in launcher


def test_runnable_examples_are_secret_free_and_path_relative() -> None:
    example_paths = [
        ROOT / "configs" / "collection.actual-only.48workers.example.json",
        ROOT / "configs" / "training.sft-only.example.json",
        ROOT / "configs" / "training.sft-alm.example.json",
        ROOT / "examples" / "collection_48workers" / "start_local.ps1",
        ROOT / "examples" / "collection_48workers" / "monitor_local.ps1",
        ROOT / "examples" / "collection_48workers" / "stop_local.ps1",
        ROOT / "examples" / "training" / "launch_pair.sh",
        ROOT / "examples" / "training" / "launch_qwen3_0_6b_full_pair.sh",
        ROOT / "examples" / "training" / "monitor_training.sh",
    ]
    forbidden = (
        "sk" + "-",
        "C:" + "\\Users\\",
        "/root/" + "autodl-tmp",
        "connect." + "weste." + "seetacloud.com",
    )
    for path in example_paths:
        text = path.read_text(encoding="utf-8")
        assert not any(value in text for value in forbidden), path


def test_repository_has_no_sensitive_literals_or_credentials() -> None:
    forbidden = (
        "C:" + "\\Users\\",
        "/root/" + "autodl-tmp",
        "connect." + "weste." + "seetacloud.com",
        "281" + "82",
        "tang" + "zhehao",
        "TANGZH" + "~1",
        "VecTo" + "RoTceV",
        "deepseek-qwen-" + "offline-alm",
        "SEE" + "TACLOUD_PASSWORD",
    )
    credential_patterns = (
        re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
        re.compile(r"\bhf_[A-Za-z0-9]{16,}\b"),
        re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}\b"),
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        re.compile(r"BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY"),
    )
    excluded_parts = {".git", "__pycache__", ".pytest_cache"}

    for path in ROOT.rglob("*"):
        if not path.is_file() or excluded_parts.intersection(path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        assert not any(value in text for value in forbidden), path.relative_to(ROOT)
        assert not any(pattern.search(text) for pattern in credential_patterns), (
            path.relative_to(ROOT)
        )


def test_repository_has_no_sensitive_filenames() -> None:
    forbidden_names = {
        ".env",
        "credentials.json",
        "id_ed25519",
        "id_rsa",
        "secrets.json",
    }
    forbidden_suffixes = {".key", ".p12", ".pem"}
    excluded_parts = {".git", "__pycache__", ".pytest_cache"}

    for path in ROOT.rglob("*"):
        if not path.is_file() or excluded_parts.intersection(path.parts):
            continue
        assert path.name.lower() not in forbidden_names, path.relative_to(ROOT)
        assert path.suffix.lower() not in forbidden_suffixes, path.relative_to(ROOT)


def test_password_ssh_deployer_uses_a_provider_neutral_environment_name() -> None:
    expected = "REMOTE_SSH_PASSWORD"
    provider_specific = "SEE" + "TACLOUD_PASSWORD"
    paths = (
        ROOT / ".env.example",
        ROOT / "README.md",
        ROOT / "scripts" / "deploy_hard_training.py",
    )

    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert expected in text, path.relative_to(ROOT)
        assert provider_specific not in text, path.relative_to(ROOT)


def test_distribution_metadata_uses_standalone_name() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'name = "deepseek-qwen-alm-toolkit"' in pyproject
