#!/usr/bin/env python3
"""Freeze the base/checkpoint candidate order for the benchmark run."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_checkpoints(train_root: Path) -> list[dict[str, Any]]:
    output_root = train_root / "outputs" / "alpha10"
    checkpoints: list[dict[str, Any]] = []
    for path in output_root.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        suffix = path.name.removeprefix("checkpoint-")
        if not suffix.isdigit():
            continue
        step = int(suffix)
        adapter = path / "adapter_model.safetensors"
        state_path = path / "trainer_state.json"
        if not adapter.is_file() or not state_path.is_file():
            raise ValueError(f"incomplete checkpoint: {path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("global_step") != step:
            raise ValueError(
                f"trainer global_step does not match {path.name}: "
                f"{state.get('global_step')!r}"
            )
        epoch = state.get("epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, (int, float)):
            raise ValueError(f"checkpoint has no numeric epoch: {path}")
        checkpoints.append(
            {
                "name": f"qwen25_instruct_lora_step_{step}",
                "role": "lora_checkpoint",
                "step": step,
                "epoch": float(epoch),
                "path": path.as_posix(),
                "adapter_sha256": sha256_file(adapter),
                "adapter_bytes": adapter.stat().st_size,
            }
        )
    checkpoints.sort(key=lambda item: item["step"])
    if not checkpoints:
        raise ValueError(f"no complete checkpoints under {output_root}")
    return checkpoints


def comparisons_for(candidate_order: list[str]) -> list[dict[str, str]]:
    comparisons = [
        {"name": f"base_vs_{name}", "old": candidate_order[0], "new": name}
        for name in candidate_order[1:]
    ]
    comparisons.extend(
        {
            "name": f"{old}_vs_{new}",
            "old": old,
            "new": new,
        }
        for old, new in zip(candidate_order[1:-1], candidate_order[2:])
    )
    return comparisons


def write_json_if_compatible(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(f"existing manifest differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_benchmark_manifests(
    plan: dict[str, Any], run_root: Path, lcb_selected_dataset: Path
) -> None:
    candidate_order = plan["candidate_order"]
    candidates = plan["candidates"]
    comparisons = comparisons_for(candidate_order)
    series_common = {
        name: {
            "role": candidates[name]["role"],
            **(
                {"adapter_sha256": candidates[name]["adapter_sha256"]}
                if candidates[name]["role"] == "lora_checkpoint"
                else {}
            ),
        }
        for name in candidate_order
    }

    human_series = {}
    for name in candidate_order:
        root = run_root / "humaneval" / "candidates" / name / "evalplus"
        human_series[name] = {
            **series_common[name],
            "raw_path": (
                root / "results" / "humaneval" / "model_hf_temp_0.0.raw.jsonl"
            ).as_posix(),
            "eval_path": (
                root
                / "results"
                / "humaneval"
                / "model_hf_temp_0.0.eval_results.json"
            ).as_posix(),
        }
    human = {
        "schema_version": "offline_alm.qwen25_instruct_humaneval_manifest.v1",
        "run_id": "qwen25_instruct_base_checkpoint_humaneval_v1_20260804",
        "dataset": {"name": "HumanEval+", "version": "v0.1.10", "tasks": 164},
        "tokenizer": {
            "model": plan["model"]["id"],
            "revision": plan["model"]["revision"],
            "path": plan["model"]["path"],
        },
        "generation": {"max_new_tokens": 768},
        "series_order": candidate_order,
        "series": human_series,
        "comparisons": comparisons,
    }
    write_json_if_compatible(run_root / "humaneval_manifest.json", human)

    dataset = {
        "name": "livecodebench/code_generation_lite",
        "release": "release_v6",
        "revision": "0fe84c3912ea0c4d4a78037083943e8f0c4dd505",
        "date_range_inclusive": ["2024-10-01", "2025-04-30"],
        "tasks": 339,
        "selected_dataset_path": lcb_selected_dataset.as_posix(),
        "selected_dataset_sha256": (
            "d7b9d4fb14931533c9b0f0be0577c27a912d4512e65b072899364a450ab5b751"
        ),
    }
    for mode in ("strict", "compatible"):
        lcb_series = {}
        for name in candidate_order:
            root = run_root / "livecodebench" / "candidates" / name
            lcb_series[name] = {
                **series_common[name],
                "generation_path": (
                    root / "strict" / "results" / "generations.jsonl"
                ).as_posix(),
                "evaluation_path": (
                    root
                    / mode
                    / "results"
                    / "codegeneration_1_0.0_full_eval_all.json"
                ).as_posix(),
            }
        lcb = {
            "schema_version": (
                f"offline_alm.qwen25_instruct_livecodebench_{mode}_manifest.v1"
            ),
            "run_id": f"qwen25_instruct_base_checkpoint_lcb_{mode}_v1_20260804",
            "dataset": dataset,
            "series_order": candidate_order,
            "series": lcb_series,
            "comparisons": comparisons,
        }
        write_json_if_compatible(
            run_root / f"livecodebench_{mode}_manifest.json", lcb
        )


def write_plan(
    *,
    train_root: Path,
    run_root: Path,
    model_dir: Path,
    model_id: str,
    model_revision: str,
    lcb_selected_dataset: Path,
) -> dict[str, Any]:
    if not (train_root / "training.completed_at").is_file():
        raise ValueError("training.completed_at is missing")
    checkpoints = discover_checkpoints(train_root)
    payload = {
        "schema_version": "offline_alm.qwen25_instruct_benchmark_plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "train_root": train_root.as_posix(),
        "run_root": run_root.as_posix(),
        "model": {
            "id": model_id,
            "revision": model_revision,
            "path": model_dir.as_posix(),
        },
        "candidate_order": ["base_qwen25_instruct"]
        + [item["name"] for item in checkpoints],
        "candidates": {
            "base_qwen25_instruct": {
                "name": "base_qwen25_instruct",
                "role": "base_model",
                "path": model_dir.as_posix(),
            },
            **{item["name"]: item for item in checkpoints},
        },
        "benchmarks": ["humaneval_plus", "livecodebench_strict", "livecodebench_compatible"],
        "automatic_shutdown": False,
        "shutdown_policy": "manual cloud control-plane shutdown required",
    }
    output = run_root / "candidate_plan.json"
    run_root.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        comparable = dict(payload)
        comparable["created_at"] = existing.get("created_at")
        if existing != comparable:
            raise ValueError("existing candidate plan differs from discovered checkpoints")
        write_benchmark_manifests(existing, run_root, lcb_selected_dataset)
        return existing
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_benchmark_manifests(payload, run_root, lcb_selected_dataset)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--lcb-selected-dataset", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = write_plan(
        train_root=args.train_root,
        run_root=args.run_root,
        model_dir=args.model_dir,
        model_id=args.model_id,
        model_revision=args.model_revision,
        lcb_selected_dataset=args.lcb_selected_dataset,
    )
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
