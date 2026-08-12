from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


RECOVERY_MODES = ("raw_fallback", "interface_wrapper")


@dataclass(frozen=True)
class RecoveryResult:
    code: str
    strategy: str


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parseable_nonempty(source: str) -> bool:
    if not source.strip():
        return False
    try:
        ast.parse(source)
    except SyntaxError:
        return False
    return True


def strip_one_boundary_fence(source: str) -> str | None:
    lines = source.strip().splitlines()
    fence_indexes = [
        index for index, line in enumerate(lines) if line.strip().startswith("```")
    ]
    if len(fence_indexes) != 1:
        return None
    index = fence_indexes[0]
    if index == 0:
        candidate = "\n".join(lines[1:]).strip()
    elif index == len(lines) - 1:
        candidate = "\n".join(lines[:-1]).strip()
    else:
        return None
    return candidate if parseable_nonempty(candidate) else None


def starter_method_name(starter_code: str) -> str | None:
    match = re.search(
        r"^\s*def\s+([A-Za-z_]\w*)\s*\(",
        starter_code,
        flags=re.MULTILINE,
    )
    return match.group(1) if match else None


def first_argument_name(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    positional = [*node.args.posonlyargs, *node.args.args]
    return positional[0].arg if positional else None


def wrap_leetcode_interface(source: str, starter_code: str) -> str | None:
    expected_name = starter_method_name(starter_code)
    if expected_name is None:
        return None
    tree = ast.parse(source)
    if any(
        isinstance(node, ast.ClassDef) and node.name == "Solution"
        for node in tree.body
    ):
        return None

    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    targets = [node for node in functions if node.name == expected_name]
    if len(targets) != 1:
        return None

    methods = [
        node
        for node in functions
        if node is targets[0] or first_argument_name(node) == "self"
    ]
    method_ids = {id(node) for node in methods}
    class_node = ast.ClassDef(
        name="Solution",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    replacement_index = min(tree.body.index(node) for node in methods)
    rebuilt: list[ast.stmt] = []
    for index, node in enumerate(tree.body):
        if index == replacement_index:
            rebuilt.append(class_node)
        if id(node) not in method_ids:
            rebuilt.append(node)
    tree.body = rebuilt
    ast.fix_missing_locations(tree)
    wrapped = ast.unparse(tree).strip()
    return wrapped if parseable_nonempty(wrapped) else None


def recover_code(
    generation: dict[str, Any],
    problem: dict[str, Any],
    *,
    mode: str,
) -> RecoveryResult:
    if mode not in RECOVERY_MODES:
        raise ValueError(f"unsupported recovery mode: {mode}")

    official = str(generation.get("extracted_code") or "").strip()
    if official:
        return RecoveryResult(official, "official")

    raw = str(generation.get("output_text") or "").strip()
    candidate = raw if parseable_nonempty(raw) else strip_one_boundary_fence(raw)
    if candidate is None:
        return RecoveryResult("", "unrecovered")

    strategy = "raw_fallback" if candidate == raw else "single_fence"
    platform = str(generation.get("platform") or problem.get("platform") or "")
    if mode == "interface_wrapper" and platform.lower() == "leetcode":
        wrapped = wrap_leetcode_interface(
            candidate,
            str(problem.get("starter_code") or ""),
        )
        if wrapped is not None:
            return RecoveryResult(wrapped, "leetcode_interface_wrapper")
    return RecoveryResult(candidate, strategy)


def unique_by_id(
    rows: Iterable[dict[str, Any]],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        question_id = str(row["question_id"])
        if question_id in result:
            raise ValueError(f"duplicate {label} question_id: {question_id}")
        result[question_id] = row
    return result


def recover_document(
    strict_rows: list[dict[str, Any]],
    generation_rows: list[dict[str, Any]],
    *,
    mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    problems = unique_by_id(strict_rows, label="strict result")
    generations = unique_by_id(generation_rows, label="generation")
    if set(problems) != set(generations):
        missing = sorted(set(problems) - set(generations))[:10]
        extra = sorted(set(generations) - set(problems))[:10]
        raise ValueError(
            "strict results and generations must contain the exact task IDs: "
            f"missing={missing} extra={extra}"
        )

    recovered_rows: list[dict[str, Any]] = []
    strategies: Counter[str] = Counter()
    platform_strategies: dict[str, Counter[str]] = {}
    per_task: list[dict[str, Any]] = []
    ast_parseable = 0
    empty = 0

    for strict in strict_rows:
        question_id = str(strict["question_id"])
        generation = generations[question_id]
        output_list = strict.get("output_list")
        code_list = strict.get("code_list")
        if not isinstance(output_list, list) or len(output_list) != 1:
            raise ValueError(f"{question_id} must have exactly one output_list item")
        if not isinstance(code_list, list) or len(code_list) != 1:
            raise ValueError(f"{question_id} must have exactly one code_list item")
        if str(output_list[0]) != str(generation.get("output_text") or ""):
            raise ValueError(f"raw output drift for {question_id}")
        if str(code_list[0]) != str(generation.get("extracted_code") or ""):
            raise ValueError(f"strict extraction drift for {question_id}")

        strict_platform = str(strict.get("platform") or "").lower()
        generation_platform = str(generation.get("platform") or "").lower()
        if strict_platform and generation_platform != strict_platform:
            raise ValueError(f"platform drift for {question_id}")

        result = recover_code(generation, strict, mode=mode)
        recovered = copy.deepcopy(strict)
        recovered["code_list"] = [result.code]
        recovered_rows.append(recovered)

        platform = generation_platform or strict_platform or "unknown"
        strategies[result.strategy] += 1
        platform_strategies.setdefault(platform, Counter())[result.strategy] += 1
        parsed = parseable_nonempty(result.code)
        ast_parseable += parsed
        empty += not bool(result.code.strip())
        per_task.append(
            {
                "question_id": question_id,
                "platform": platform,
                "strategy": result.strategy,
                "strict_empty": not bool(str(code_list[0]).strip()),
                "recovered_empty": not bool(result.code.strip()),
                "recovered_ast_parseable": parsed,
                "raw_sha256": sha256_bytes(
                    str(output_list[0]).encode("utf-8")
                ),
                "recovered_code_sha256": sha256_bytes(
                    result.code.encode("utf-8")
                ),
            }
        )

    audit = {
        "mode": mode,
        "records": len(recovered_rows),
        "strategy_counts": dict(sorted(strategies.items())),
        "platform_strategy_counts": {
            platform: dict(sorted(counts.items()))
            for platform, counts in sorted(platform_strategies.items())
        },
        "empty_after_recovery": empty,
        "ast_parseable_after_recovery": ast_parseable,
        "per_task": per_task,
    }
    return recovered_rows, audit


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recover non-fenced LiveCodeBench code without regeneration."
    )
    parser.add_argument("--strict-results", type=Path, required=True)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--mode", choices=RECOVERY_MODES, required=True)
    parser.add_argument("--output-results", type=Path, required=True)
    parser.add_argument("--output-audit", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for output in (args.output_results, args.output_audit):
        if output.exists() and not args.force:
            raise FileExistsError(f"refusing to overwrite existing output: {output}")

    strict_rows = json.loads(args.strict_results.read_text(encoding="utf-8"))
    if not isinstance(strict_rows, list):
        raise ValueError("strict results must be a JSON list")
    generation_rows = read_jsonl(args.generations)
    recovered, audit = recover_document(
        strict_rows,
        generation_rows,
        mode=args.mode,
    )

    args.output_results.parent.mkdir(parents=True, exist_ok=True)
    args.output_audit.parent.mkdir(parents=True, exist_ok=True)
    args.output_results.write_text(
        json.dumps(recovered, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    audit.update(
        {
            "schema_version": "offline_alm.livecodebench_format_recovery.v1",
            "created_at": now_iso(),
            "strict_results_path": str(args.strict_results),
            "strict_results_sha256": sha256_file(args.strict_results),
            "generations_path": str(args.generations),
            "generations_sha256": sha256_file(args.generations),
            "output_results_path": str(args.output_results),
            "output_results_sha256": sha256_file(args.output_results),
        }
    )
    args.output_audit.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "mode": args.mode,
                "records": audit["records"],
                "strategy_counts": audit["strategy_counts"],
                "empty_after_recovery": audit["empty_after_recovery"],
                "output_results": str(args.output_results),
                "output_audit": str(args.output_audit),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
