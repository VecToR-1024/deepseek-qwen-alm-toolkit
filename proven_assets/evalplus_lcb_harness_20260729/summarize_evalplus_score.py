from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    base_passes = sum(row["base_status"] == "pass" for row in rows)
    plus_passes = sum(
        row["base_status"] == "pass" and row["plus_status"] == "pass"
        for row in rows
    )
    return {
        "tasks": len(rows),
        "base_passes": base_passes,
        "plus_passes": plus_passes,
        "base_pass_at_1": base_passes / len(rows),
        "plus_pass_at_1": plus_passes / len(rows),
    }


def summarize_evalplus_output(
    output_path: Path,
    *,
    expected_task_ids: list[str],
    heldout_task_ids: list[str] | None = None,
) -> dict[str, Any]:
    data = json.loads(output_path.read_text(encoding="utf-8"))
    expected = set(expected_task_ids)
    if len(expected) != len(expected_task_ids) or set(data["eval"]) != expected:
        raise ValueError("scored task IDs do not exactly match the frozen task set")
    rows_by_id: dict[str, dict[str, Any]] = {}
    for task_id, task_rows in data["eval"].items():
        if len(task_rows) != 1:
            raise ValueError(f"{task_id} has {len(task_rows)} scored samples, expected 1")
        rows_by_id[task_id] = task_rows[0]
    rows = [rows_by_id[task_id] for task_id in expected_task_ids]
    full = score_subset(rows)
    official = data["pass_at_k"]
    if not math.isclose(full["base_pass_at_1"], official["base"]["pass@1"]):
        raise ValueError("computed base pass@1 disagrees with EvalPlus")
    if not math.isclose(full["plus_pass_at_1"], official["plus"]["pass@1"]):
        raise ValueError("computed plus pass@1 disagrees with EvalPlus")

    summary: dict[str, Any] = {
        "dataset_hash_md5": data["hash"],
        "full": full,
        "base_status_counts": dict(
            sorted(collections.Counter(row["base_status"] for row in rows).items())
        ),
        "plus_status_counts": dict(
            sorted(collections.Counter(row["plus_status"] for row in rows).items())
        ),
    }
    if heldout_task_ids is not None:
        if not set(heldout_task_ids).issubset(expected):
            raise ValueError("held-out task IDs are not a subset of scored tasks")
        summary["heldout"] = score_subset(
            [rows_by_id[task_id] for task_id in heldout_task_ids]
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--dataset", choices=("humaneval", "mbpp"), required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    codegen_summary_path = (
        args.candidate_root / "logs" / f"{args.dataset}_codegen_summary.json"
    )
    codegen = json.loads(codegen_summary_path.read_text(encoding="utf-8"))
    result_path = (
        args.candidate_root
        / "results"
        / args.dataset
        / "model_hf_temp_0.0.eval_results.json"
    )
    heldout_ids = None
    if args.dataset == "mbpp":
        if args.heldout_manifest is None:
            raise ValueError("--heldout-manifest is required for MBPP")
        heldout = json.loads(args.heldout_manifest.read_text(encoding="utf-8"))
        heldout_ids = heldout["heldout_task_ids"]
    score = summarize_evalplus_output(
        result_path,
        expected_task_ids=codegen["task_ids"],
        heldout_task_ids=heldout_ids,
    )
    summary = {
        "schema_version": "offline_alm.evalplus.score.v1",
        "candidate": args.candidate,
        "adapter": codegen["adapter"],
        "adapter_sha256": codegen["adapter_sha256"],
        "evalplus_commit": codegen["evalplus_commit"],
        "dataset": args.dataset,
        "execution": {
            "runner_user": "evalplus-runner",
            "network_filter": "libseccomp deny socket domain != AF_UNIX",
            "parallel_workers": 10,
            "base_only": False,
            "test_details": False,
            "min_time_limit": 4.0,
            "gt_time_limit_factor": 4.0,
        },
        **score,
        "result_path": str(result_path),
        "result_sha256": sha256_file(result_path),
        "codegen_summary_sha256": sha256_file(codegen_summary_path),
    }
    summary_path = args.candidate_root / "logs" / f"{args.dataset}_score_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
