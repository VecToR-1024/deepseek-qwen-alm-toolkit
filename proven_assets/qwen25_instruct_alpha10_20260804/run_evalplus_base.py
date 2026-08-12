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


def validate_model_snapshot(path: Path, revision: str) -> None:
    resolved = path.resolve()
    if resolved.name != revision:
        raise ValueError(
            f"revision mismatch: snapshot={resolved.name!r} expected={revision!r}"
        )
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        if not (resolved / name).is_file():
            raise ValueError(f"missing model snapshot file: {name}")
    if not any(resolved.glob("*.safetensors")):
        raise ValueError("model snapshot has no safetensors weights")


def validate_codegen_outputs(
    sample_path: Path,
    raw_path: Path,
    *,
    expected_task_ids: list[str],
    allow_empty_sanitized: bool = False,
) -> dict[str, int]:
    samples = read_jsonl(sample_path)
    raw = read_jsonl(raw_path)
    expected = set(expected_task_ids)
    for label, rows in (("sanitized", samples), ("raw", raw)):
        ids = [row["task_id"] for row in rows]
        if len(ids) != len(expected_task_ids) or len(ids) != len(set(ids)):
            raise ValueError(f"{label} task IDs are incomplete or duplicated")
        if set(ids) != expected:
            raise ValueError(f"{label} task IDs do not match the frozen task set")
    empty_sanitized = 0
    for row in samples:
        solution = row["solution"]
        if not isinstance(solution, str):
            raise ValueError(f"non-string sanitized solution for {row['task_id']}")
        if not solution.strip():
            empty_sanitized += 1
            if allow_empty_sanitized:
                continue
            raise ValueError(f"empty sanitized solution for {row['task_id']}")
        ast.parse(solution)
    return {
        "sanitized_records": len(samples),
        "raw_records": len(raw),
        "sanitized_ast_parse_successes": len(samples) - empty_sanitized,
        "empty_sanitized_solutions": empty_sanitized,
    }


def ensure_model_link(run_root: Path, base_model: Path) -> Path:
    link = run_root / "model"
    if link.is_symlink():
        if link.resolve() != base_model.resolve():
            raise RuntimeError(f"model link points to {link.resolve()}")
    elif link.exists():
        raise RuntimeError(f"refusing to replace non-symlink model path: {link}")
    else:
        link.symlink_to(base_model, target_is_directory=True)
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


def base_candidate_metadata(base_model: Path) -> dict[str, str | None]:
    return {
        "candidate_role": "base_model",
        "base_model": str(base_model),
        # The shared scorer predates base-model candidates and expects these
        # adapter keys. Nulls retain its schema without inventing an adapter.
        "adapter": None,
        "adapter_sha256": None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--dataset", choices=("humaneval", "mbpp"), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--id-range", type=int, nargs=2)
    parser.add_argument("--allow-empty-sanitized", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.base_model = args.base_model.resolve()
    validate_model_snapshot(args.base_model, args.model_revision)
    args.run_root.mkdir(parents=True, exist_ok=True)
    (args.run_root / "logs").mkdir(exist_ok=True)
    ensure_model_link(args.run_root, args.base_model)
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
        allow_empty_sanitized=args.allow_empty_sanitized,
    )
    summary = {
        "schema_version": "offline_alm.evalplus.base_codegen.v1",
        "candidate": args.candidate,
        **base_candidate_metadata(args.base_model),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
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
