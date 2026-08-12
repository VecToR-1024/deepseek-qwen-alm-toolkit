from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.audit_training_run import build_report


REQUIRED_CHECKPOINT_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "optimizer.pt",
    "rng_state.pth",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def create_arm(
    run_root: Path,
    arm: str,
    *,
    alpha_alm: float,
    adapter_bytes: bytes,
) -> None:
    log_dir = run_root / "logs" / arm
    output_dir = run_root / "outputs" / arm
    write_text(log_dir / "started_at", "2026-07-29T10:00:00+08:00\n")
    write_text(log_dir / "finished_at", "2026-07-29T10:01:00+08:00\n")
    write_text(log_dir / "exit_code", "0\n")
    write_text(
        log_dir / "gpu_samples.csv",
        "2026/07/29 10:00:00, 100, 50\n2026/07/29 10:00:01, 200, 100\n",
    )
    config = {
        "event": "offline_alm_training_start",
        "alpha_alm": alpha_alm,
        "dataset_records": 4,
        "output_dir": str(output_dir),
        "seed": 7,
    }
    write_text(log_dir / "train.log", json.dumps(config) + "\n")

    rows = []
    for step in range(1, 5):
        row = {
            "step": step,
            "epoch": step / 2,
            "hard_sft_loss": 1.0 / step,
            "combined_loss": 1.0 / step + alpha_alm * 0.01,
            "grad_norm": float(step),
            "learning_rate": 0.001,
        }
        if alpha_alm:
            row["alm_loss"] = 0.01
            row["weighted_alm_loss"] = 0.01
        rows.append(row)
    rows.append(
        {
            "step": 4,
            "epoch": 2.0,
            "train_runtime": 60.0,
            "train_loss": 0.5,
        }
    )
    trainer_state = {"epoch": 2.0, "global_step": 4, "log_history": rows}

    for checkpoint_id in (2, 3, 4):
        checkpoint = output_dir / f"checkpoint-{checkpoint_id}"
        checkpoint.mkdir(parents=True)
        for filename in REQUIRED_CHECKPOINT_FILES:
            if filename == "trainer_state.json":
                write_text(checkpoint / filename, json.dumps(trainer_state))
            elif filename == "adapter_model.safetensors":
                (checkpoint / filename).write_bytes(
                    adapter_bytes if checkpoint_id == 4 else b"earlier"
                )
            else:
                write_text(checkpoint / filename, filename)
    (output_dir / "adapter_model.safetensors").write_bytes(adapter_bytes)


def test_build_report_accepts_complete_controlled_pair(tmp_path: Path) -> None:
    write_text(tmp_path / "launcher_exit_code", "0\n")
    write_text(
        tmp_path / "training_manifest.json",
        json.dumps(
            {
                "training_code": {"authoritative_commit": "abc123"},
                "dataset": {"sha256": "data-sha", "records": 4},
            }
        ),
    )
    create_arm(tmp_path, "sft_only", alpha_alm=0.0, adapter_bytes=b"sft")
    create_arm(tmp_path, "sft_alm", alpha_alm=1.0, adapter_bytes=b"alm")

    report = build_report(
        tmp_path,
        expected_steps=4,
        expected_epochs=2.0,
        expected_checkpoint_ids=[2, 3, 4],
    )

    assert report["status"] == "passed"
    assert report["controlled_fields_match"] is True
    assert report["alpha_is_only_objective_difference"] is True
    assert report["initial_hard_sft_loss_match"] is True
    assert report["arms"]["sft_only"]["step_rows"] == 4
    assert report["arms"]["sft_only"]["all_metrics_finite"] is True
    assert report["arms"]["sft_only"]["gpu_peak_memory_mib"] == 200
    assert report["arms"]["sft_alm"]["checkpoints"]["4"]["complete"] is True
    assert report["arms"]["sft_alm"]["root_matches_final_checkpoint"] is True
    assert report["arms"]["sft_only"]["root_adapter_sha256"] == hashlib.sha256(
        b"sft"
    ).hexdigest()


def test_build_report_rejects_non_finite_or_missing_checkpoint(
    tmp_path: Path,
) -> None:
    write_text(tmp_path / "launcher_exit_code", "0\n")
    write_text(
        tmp_path / "training_manifest.json",
        json.dumps(
            {
                "training_code": {"authoritative_commit": "abc123"},
                "dataset": {"sha256": "data-sha", "records": 4},
            }
        ),
    )
    create_arm(tmp_path, "sft_only", alpha_alm=0.0, adapter_bytes=b"sft")
    create_arm(tmp_path, "sft_alm", alpha_alm=1.0, adapter_bytes=b"alm")

    state_path = (
        tmp_path
        / "outputs"
        / "sft_alm"
        / "checkpoint-4"
        / "trainer_state.json"
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["log_history"][0]["combined_loss"] = float("nan")
    state_path.write_text(json.dumps(state), encoding="utf-8")
    (
        tmp_path
        / "outputs"
        / "sft_alm"
        / "checkpoint-3"
        / "optimizer.pt"
    ).unlink()

    report = build_report(
        tmp_path,
        expected_steps=4,
        expected_epochs=2.0,
        expected_checkpoint_ids=[2, 3, 4],
    )

    assert report["status"] == "failed"
    assert report["arms"]["sft_alm"]["all_metrics_finite"] is False
    assert report["arms"]["sft_alm"]["checkpoints"]["3"]["complete"] is False
