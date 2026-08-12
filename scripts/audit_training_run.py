from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any


ARMS = ("sft_only", "sft_alm")
REQUIRED_CHECKPOINT_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "optimizer.pt",
    "rng_state.pth",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
)
METRIC_FIELDS = (
    "loss",
    "hard_sft_loss",
    "alm_loss",
    "weighted_alm_loss",
    "combined_loss",
    "grad_norm",
    "learning_rate",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def seconds_between(start: str, finish: str) -> float:
    return (datetime.fromisoformat(finish) - datetime.fromisoformat(start)).total_seconds()


def find_start_config(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for match in re.finditer(r"\{[^{}\r\n]+\}", text):
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if value.get("event") == "offline_alm_training_start":
            return value
    raise ValueError(f"training start event not found in {log_path}")


def checkpoint_audit(path: Path) -> dict[str, Any]:
    missing = [
        filename for filename in REQUIRED_CHECKPOINT_FILES if not (path / filename).is_file()
    ]
    adapter = path / "adapter_model.safetensors"
    return {
        "complete": not missing,
        "missing_files": missing,
        "adapter_sha256": sha256_file(adapter) if adapter.is_file() else None,
    }


def gpu_peak_memory_mib(path: Path) -> int | None:
    if not path.is_file():
        return None
    values = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            values.append(int(parts[1]))
        except ValueError:
            continue
    return max(values) if values else None


def average(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return statistics.fmean(values) if values else None


def epoch_averages(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        epoch = max(1, math.ceil(float(row["epoch"]) - 1e-9))
        grouped.setdefault(epoch, []).append(row)
    return [
        {
            "epoch": epoch,
            "hard_sft_loss_mean": average(epoch_rows, "hard_sft_loss"),
            "alm_loss_mean": average(epoch_rows, "alm_loss"),
            "combined_loss_mean": average(epoch_rows, "combined_loss"),
        }
        for epoch, epoch_rows in sorted(grouped.items())
    ]


def audit_arm(
    run_root: Path,
    arm: str,
    *,
    expected_steps: int,
    expected_epochs: float,
    expected_checkpoint_ids: list[int],
) -> dict[str, Any]:
    log_dir = run_root / "logs" / arm
    output_dir = run_root / "outputs" / arm
    final_checkpoint = output_dir / f"checkpoint-{expected_steps}"
    state = json.loads((final_checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    step_rows = [
        row for row in state["log_history"] if "hard_sft_loss" in row and "step" in row
    ]
    trainer_final_rows = [
        row for row in state["log_history"] if "train_runtime" in row
    ]
    observed_steps = [int(row["step"]) for row in step_rows]
    expected_step_sequence = list(range(1, expected_steps + 1))

    non_finite = []
    for row in step_rows:
        for field in METRIC_FIELDS:
            value = row.get(field)
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                non_finite.append({"step": row["step"], "field": field, "value": value})

    checkpoints = {
        str(checkpoint_id): checkpoint_audit(
            output_dir / f"checkpoint-{checkpoint_id}"
        )
        for checkpoint_id in expected_checkpoint_ids
    }
    root_adapter = output_dir / "adapter_model.safetensors"
    root_hash = sha256_file(root_adapter) if root_adapter.is_file() else None
    final_hash = checkpoints[str(expected_steps)]["adapter_sha256"]
    started_at = read_text(log_dir / "started_at")
    finished_at = read_text(log_dir / "finished_at")
    exit_code = int(read_text(log_dir / "exit_code"))
    config = find_start_config(log_dir / "train.log")

    checks = {
        "exit_code_zero": exit_code == 0,
        "global_step_matches": int(state["global_step"]) == expected_steps,
        "epoch_matches": math.isclose(float(state["epoch"]), expected_epochs),
        "step_sequence_contiguous": observed_steps == expected_step_sequence,
        "all_metrics_finite": not non_finite,
        "all_checkpoints_complete": all(
            checkpoint["complete"] for checkpoint in checkpoints.values()
        ),
        "root_matches_final_checkpoint": root_hash == final_hash,
    }
    return {
        "arm": arm,
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "exit_code": exit_code,
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_clock_seconds": seconds_between(started_at, finished_at),
        "config": config,
        "step_rows": len(step_rows),
        "global_step": int(state["global_step"]),
        "all_metrics_finite": not non_finite,
        "non_finite_metrics": non_finite,
        "max_grad_norm": max(float(row["grad_norm"]) for row in step_rows),
        "gpu_peak_memory_mib": gpu_peak_memory_mib(log_dir / "gpu_samples.csv"),
        "first_5_average": {
            "hard_sft_loss": average(step_rows[:5], "hard_sft_loss"),
            "alm_loss": average(step_rows[:5], "alm_loss"),
            "combined_loss": average(step_rows[:5], "combined_loss"),
        },
        "last_5_average": {
            "hard_sft_loss": average(step_rows[-5:], "hard_sft_loss"),
            "alm_loss": average(step_rows[-5:], "alm_loss"),
            "combined_loss": average(step_rows[-5:], "combined_loss"),
        },
        "first_step": step_rows[0],
        "final_step": step_rows[-1],
        "trainer_final": trainer_final_rows[-1] if trainer_final_rows else None,
        "epoch_averages": epoch_averages(step_rows),
        "checkpoint_ids": expected_checkpoint_ids,
        "checkpoints": checkpoints,
        "root_adapter_sha256": root_hash,
        "root_matches_final_checkpoint": root_hash == final_hash,
    }


def comparable_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in config.items()
        if key not in {"alpha_alm", "output_dir"}
    }


def build_report(
    run_root: Path,
    *,
    expected_steps: int,
    expected_epochs: float,
    expected_checkpoint_ids: list[int],
) -> dict[str, Any]:
    manifest = json.loads(
        (run_root / "training_manifest.json").read_text(encoding="utf-8")
    )
    arms = {
        arm: audit_arm(
            run_root,
            arm,
            expected_steps=expected_steps,
            expected_epochs=expected_epochs,
            expected_checkpoint_ids=expected_checkpoint_ids,
        )
        for arm in ARMS
    }
    sft = arms["sft_only"]
    alm = arms["sft_alm"]
    controlled_fields_match = comparable_config(sft["config"]) == comparable_config(
        alm["config"]
    )
    alpha_only = (
        controlled_fields_match
        and float(sft["config"]["alpha_alm"]) == 0.0
        and float(alm["config"]["alpha_alm"]) == 1.0
    )
    initial_hard_match = sft["first_step"]["hard_sft_loss"] == alm["first_step"][
        "hard_sft_loss"
    ]
    launcher_exit_code = int(read_text(run_root / "launcher_exit_code"))
    top_checks = {
        "launcher_exit_code_zero": launcher_exit_code == 0,
        "controlled_fields_match": controlled_fields_match,
        "alpha_is_only_objective_difference": alpha_only,
        "initial_hard_sft_loss_match": initial_hard_match,
        "all_arms_passed": all(arm["status"] == "passed" for arm in arms.values()),
        "final_adapters_are_distinct": (
            sft["root_adapter_sha256"] != alm["root_adapter_sha256"]
        ),
    }
    return {
        "schema_version": "offline_alm.training_audit.v1",
        "status": "passed" if all(top_checks.values()) else "failed",
        **top_checks,
        "code_commit": manifest["training_code"]["authoritative_commit"],
        "training_data_sha256": manifest["dataset"]["sha256"],
        "training_records": manifest["dataset"]["records"],
        "expected_steps": expected_steps,
        "expected_epochs": expected_epochs,
        "expected_checkpoint_ids": expected_checkpoint_ids,
        "arms": arms,
        "comparison": {
            "runtime_ratio_alm_over_sft": (
                alm["wall_clock_seconds"] / sft["wall_clock_seconds"]
            ),
            "final_adapters_are_distinct": top_checks[
                "final_adapters_are_distinct"
            ],
        },
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Offline ALM training audit",
        "",
        f"- Status: `{report['status']}`",
        f"- Records: `{report['training_records']}`",
        f"- Data SHA-256: `{report['training_data_sha256']}`",
        f"- Code commit: `{report['code_commit']}`",
        "",
        "| Arm | Status | Steps | Wall seconds | Peak MiB | Final adapter SHA-256 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for arm_name in ARMS:
        arm = report["arms"][arm_name]
        lines.append(
            f"| {arm_name} | {arm['status']} | {arm['global_step']} | "
            f"{arm['wall_clock_seconds']:.0f} | {arm['gpu_peak_memory_mib']} | "
            f"`{arm['root_adapter_sha256']}` |"
        )
    lines.extend(
        [
            "",
            f"- Controlled fields match: `{report['controlled_fields_match']}`",
            "- Alpha is the only intended objective difference: "
            f"`{report['alpha_is_only_objective_difference']}`",
            f"- Initial hard-SFT loss matches: `{report['initial_hard_sft_loss_match']}`",
            "- ALM/SFT wall-clock ratio: "
            f"`{report['comparison']['runtime_ratio_alm_over_sft']:.6f}`",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--expected-epochs", type=float, required=True)
    parser.add_argument("--expected-checkpoints", type=int, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_json = args.output_json or args.run_root / "training_audit.json"
    output_md = args.output_md or args.run_root / "training_audit.md"
    if not args.force and (output_json.exists() or output_md.exists()):
        raise FileExistsError("refusing to overwrite an existing training audit")
    report = build_report(
        args.run_root,
        expected_steps=args.expected_steps,
        expected_epochs=args.expected_epochs,
        expected_checkpoint_ids=args.expected_checkpoints,
    )
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output_md.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output_json)}))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
