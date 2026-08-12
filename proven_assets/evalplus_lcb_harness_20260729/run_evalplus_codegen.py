from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any


EVALPLUS_COMMIT = "26d6d00bb1fd0fa37f39c99d5290da67891d1c5e"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_codegen_outputs(
    sample_path: Path,
    raw_path: Path,
    *,
    expected_task_ids: list[str],
) -> dict[str, int]:
    samples = read_jsonl(sample_path)
    raw = read_jsonl(raw_path)
    sample_ids = [row["task_id"] for row in samples]
    raw_ids = [row["task_id"] for row in raw]
    expected = set(expected_task_ids)
    if (
        len(sample_ids) != len(expected_task_ids)
        or len(set(sample_ids)) != len(sample_ids)
        or set(sample_ids) != expected
    ):
        raise ValueError("sanitized task IDs do not exactly match the frozen task set")
    if (
        len(raw_ids) != len(expected_task_ids)
        or len(set(raw_ids)) != len(raw_ids)
        or set(raw_ids) != expected
    ):
        raise ValueError("raw task IDs do not exactly match the frozen task set")
    for row in samples:
        ast.parse(row["solution"])
    return {
        "sanitized_records": len(samples),
        "raw_records": len(raw),
        "sanitized_ast_parse_successes": len(samples),
    }


def ensure_model_link(run_root: Path, adapter: Path) -> Path:
    link = run_root / "model"
    if link.is_symlink():
        if link.resolve() != adapter.resolve():
            raise RuntimeError(f"model link points to {link.resolve()}, not {adapter}")
    elif link.exists():
        raise RuntimeError(f"refusing to replace non-symlink model path: {link}")
    else:
        link.symlink_to(adapter, target_is_directory=True)
    return link


def expected_tasks(dataset: str, id_range: tuple[int, int] | None) -> list[str]:
    from evalplus.data import get_human_eval_plus, get_mbpp_plus

    data = get_human_eval_plus() if dataset == "humaneval" else get_mbpp_plus()
    task_ids = list(data)
    if id_range is None:
        return task_ids
    low, high = id_range
    return [
        task_id
        for task_id in task_ids
        if low <= int(task_id.split("/", 1)[1]) < high
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--adapter-sha256", required=True)
    parser.add_argument("--dataset", choices=("humaneval", "mbpp"), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--id-range", type=int, nargs=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.adapter = args.adapter.resolve()
    args.run_root.mkdir(parents=True, exist_ok=True)
    (args.run_root / "logs").mkdir(exist_ok=True)
    actual_adapter_hash = sha256_file(args.adapter / "adapter_model.safetensors")
    if actual_adapter_hash != args.adapter_sha256:
        raise RuntimeError(
            f"adapter SHA-256 mismatch: {actual_adapter_hash} != "
            f"{args.adapter_sha256}"
        )
    ensure_model_link(args.run_root, args.adapter)
    frozen_task_ids = expected_tasks(
        args.dataset,
        tuple(args.id_range) if args.id_range else None,
    )
    if not frozen_task_ids:
        raise RuntimeError("frozen EvalPlus task selection is empty")

    # Pinned official implementation:
    # https://github.com/evalplus/evalplus/blob/26d6d00bb1fd0fa37f39c99d5290da67891d1c5e/evalplus/codegen.py
    from evalplus.codegen import run_codegen

    started_at = now_iso()
    started = time.perf_counter()
    old_cwd = Path.cwd()
    try:
        os.chdir(args.run_root)
        sample_name = run_codegen(
            model="./model",
            dataset=args.dataset,
            root="results",
            bs=1,
            n_samples=1,
            temperature=0.0,
            greedy=True,
            backend="hf",
            attn_implementation="eager",
            device_map="auto",
            dtype="bfloat16",
            max_new_tokens=768,
            resume=True,
            id_range=args.id_range,
        )
    finally:
        os.chdir(old_cwd)
    elapsed = time.perf_counter() - started

    sample_path = args.run_root / sample_name
    raw_path = sample_path.with_name(sample_path.name.replace(".jsonl", ".raw.jsonl"))
    validation = validate_codegen_outputs(
        sample_path,
        raw_path,
        expected_task_ids=frozen_task_ids,
    )
    summary = {
        "schema_version": "offline_alm.evalplus.codegen.v1",
        "candidate": args.candidate,
        "adapter": str(args.adapter),
        "adapter_sha256": actual_adapter_hash,
        "evalplus_commit": EVALPLUS_COMMIT,
        "dataset": args.dataset,
        "task_ids": frozen_task_ids,
        "task_count": len(frozen_task_ids),
        "generation": {
            "backend": "hf",
            "temperature": 0.0,
            "greedy": True,
            "batch_size": 1,
            "n_samples": 1,
            "max_new_tokens": 768,
            "dtype": "bfloat16",
            "attention": "eager",
            "device_map": "auto",
            "prompt_and_sanitize": "EvalPlus official implementation",
        },
        "started_at": started_at,
        "finished_at": now_iso(),
        "elapsed_seconds": elapsed,
        **validation,
        "sample_path": str(sample_path),
        "raw_path": str(raw_path),
        "sample_sha256": sha256_file(sample_path),
        "raw_sha256": sha256_file(raw_path),
    }
    summary_path = args.run_root / "logs" / f"{args.dataset}_codegen_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
