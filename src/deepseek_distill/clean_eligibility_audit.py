"""Streaming publication of clean-training eligibility artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TextIO

from .clean_eligibility import (
    CleanEligibilityPolicy,
    evaluate_clean_eligibility,
)


ELIGIBILITY_NAME = "existing_v3_eligibility.jsonl"
RETAINED_NAME = "existing_v3_retained.jsonl"
EXCLUDED_NAME = "existing_v3_excluded.jsonl"
REPORT_JSON_NAME = "existing_v3_clean_audit.json"
REPORT_MARKDOWN_NAME = "existing_v3_clean_audit.md"
EXCLUSION_SCHEMA_VERSION = "offline_alm.clean_exclusion.v1"
AUDIT_SCHEMA_VERSION = "offline_alm.clean_eligibility_audit.v1"


def build_clean_eligibility_outputs(
    *,
    training_data: Path,
    alm_diagnostics: Path,
    eos_attestation: Path,
    output_dir: Path,
    policy: CleanEligibilityPolicy | None = None,
) -> dict[str, Any]:
    """Audit a JSONL stream and publish immutable retained/excluded indexes."""

    selected_policy = policy or CleanEligibilityPolicy()
    targets = {
        "eligibility": output_dir / ELIGIBILITY_NAME,
        "retained": output_dir / RETAINED_NAME,
        "excluded": output_dir / EXCLUDED_NAME,
        "report_json": output_dir / REPORT_JSON_NAME,
        "report_markdown": output_dir / REPORT_MARKDOWN_NAME,
    }
    existing = [path for path in targets.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing clean-audit artifacts: "
            + ", ".join(str(path) for path in existing)
        )

    training_hash = sha256_file(training_data)
    eos_records = _validate_eos_attestation(
        eos_attestation,
        training_sha256=training_hash,
    )
    diagnostics = _load_alm_diagnostics(alm_diagnostics)
    output_dir.mkdir(parents=True, exist_ok=True)

    temporary_paths = {
        name: _temporary_path(path)
        for name, path in targets.items()
    }
    counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    source_counts: dict[str, Counter[str]] = {}
    seen_ids: set[str] = set()
    try:
        with (
            training_data.open("r", encoding="utf-8") as source_handle,
            _open_text(temporary_paths["eligibility"]) as eligibility_handle,
            _open_text(temporary_paths["retained"]) as retained_handle,
            _open_text(temporary_paths["excluded"]) as excluded_handle,
        ):
            for line_number, line in enumerate(source_handle, start=1):
                if not line.strip():
                    continue
                record = _parse_record(training_data, line_number, line)
                record_id = record.get("id")
                if not isinstance(record_id, str) or not record_id:
                    raise ValueError(
                        f"{training_data}:{line_number}: id must be a non-empty string"
                    )
                if record_id in seen_ids:
                    raise ValueError(
                        f"{training_data}:{line_number}: duplicate id {record_id!r}"
                    )
                seen_ids.add(record_id)
                decision = evaluate_clean_eligibility(
                    record,
                    alm_diagnostic=diagnostics.get(record_id),
                    eos_supervised=True,
                    policy=selected_policy,
                )
                _write_json_line(eligibility_handle, decision)
                counts["total"] += 1
                source = decision["source"]
                source_counter = source_counts.setdefault(source, Counter())
                source_counter["total"] += 1
                if decision["eligible"]:
                    counts["eligible"] += 1
                    source_counter["eligible"] += 1
                    retained_handle.write(line)
                    if not line.endswith("\n"):
                        retained_handle.write("\n")
                else:
                    counts["excluded"] += 1
                    source_counter["excluded"] += 1
                    reason_counts.update(decision["reasons"])
                    _write_json_line(
                        excluded_handle,
                        {
                            "schema_version": EXCLUSION_SCHEMA_VERSION,
                            "id": record_id,
                            "source": source,
                            "reasons": decision["reasons"],
                        },
                    )
        for name in ("eligibility", "retained", "excluded"):
            _fsync_path(temporary_paths[name])
        if counts["total"] != eos_records:
            raise ValueError(
                "EOS attestation record count does not match training data: "
                f"{eos_records} != {counts['total']}"
            )

        jsonl_outputs = {
            name: _output_metadata(
                temporary_paths[name],
                targets[name],
                records=(
                    counts["total"]
                    if name == "eligibility"
                    else counts["eligible"]
                    if name == "retained"
                    else counts["excluded"]
                ),
            )
            for name in ("eligibility", "retained", "excluded")
        }
        report = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "inputs": {
                "training_data": str(training_data),
                "training_data_sha256": training_hash,
                "alm_diagnostics": str(alm_diagnostics),
                "alm_diagnostics_sha256": sha256_file(alm_diagnostics),
                "eos_attestation": str(eos_attestation),
                "eos_attestation_sha256": sha256_file(eos_attestation),
            },
            "policy": {
                "max_comment_line_ratio": selected_policy.max_comment_line_ratio,
                "max_sequence_length": selected_policy.max_sequence_length,
                "raw_trace_mutation_allowed": False,
            },
            "counts": {
                "total": counts["total"],
                "eligible": counts["eligible"],
                "excluded": counts["excluded"],
            },
            "source_counts": {
                source: {
                    "total": values["total"],
                    "eligible": values["eligible"],
                    "excluded": values["excluded"],
                }
                for source, values in sorted(source_counts.items())
            },
            "reason_counts": dict(sorted(reason_counts.items())),
            "eos_attestation": {
                "records": eos_records,
                "all_records_supervised": True,
            },
            "outputs": jsonl_outputs,
        }
        _write_text(
            temporary_paths["report_json"],
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        )
        _write_text(
            temporary_paths["report_markdown"],
            render_clean_audit_markdown(report),
        )
        for name, target in targets.items():
            os.replace(temporary_paths[name], target)
        return report
    except BaseException:
        for temporary in temporary_paths.values():
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
        raise


def render_clean_audit_markdown(report: Mapping[str, Any]) -> str:
    counts = report["counts"]
    lines = [
        "# Existing v3 clean-training eligibility audit",
        "",
        "## Outcome",
        "",
        f"- Total records: {counts['total']}",
        f"- Eligible: {counts['eligible']}",
        f"- Excluded: {counts['excluded']}",
        "- Raw teacher responses and traces modified: 0",
        "",
        "## Source counts",
        "",
        "| Source | Total | Eligible | Excluded |",
        "|---|---:|---:|---:|",
    ]
    for source, values in report["source_counts"].items():
        lines.append(
            f"| {source} | {values['total']} | "
            f"{values['eligible']} | {values['excluded']} |"
        )
    lines.extend(
        [
            "",
            "## Exclusion reasons",
            "",
            "| Reason | Count |",
            "|---|---:|",
        ]
    )
    for reason, count in report["reason_counts"].items():
        lines.append(f"| `{reason}` | {count} |")
    lines.extend(
        [
            "",
            "The retained JSONL preserves complete original records. The excluded "
            "JSONL is a lightweight index; original records remain in the immutable "
            "input dataset.",
            "",
        ]
    )
    return "\n".join(lines)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_eos_attestation(path: Path, *, training_sha256: str) -> int:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("EOS attestation must be a JSON object")
    inputs = value.get("inputs")
    training = value.get("training_contract")
    if not isinstance(inputs, Mapping) or not isinstance(training, Mapping):
        raise ValueError("EOS attestation is missing inputs/training_contract")
    if inputs.get("training_data_sha256") != training_sha256:
        raise ValueError("EOS attestation SHA-256 does not match training data")
    ends = training.get("end_token_supervision")
    if not isinstance(ends, Mapping):
        raise ValueError("EOS attestation is missing end_token_supervision")
    records = training.get("records")
    supervised = ends.get("eos_supervised_records")
    if (
        isinstance(records, bool)
        or not isinstance(records, int)
        or records <= 0
        or supervised != records
    ):
        raise ValueError("EOS attestation does not supervise every record")
    return records


def _load_alm_diagnostics(path: Path) -> dict[str, dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or not isinstance(value.get("examples"), list):
        raise ValueError("ALM diagnostics must contain an examples list")
    diagnostics: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(value["examples"]):
        if not isinstance(item, dict):
            raise ValueError(f"ALM diagnostics examples[{index}] must be an object")
        record_id = item.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"ALM diagnostics examples[{index}] has invalid id")
        if record_id in diagnostics:
            raise ValueError(f"duplicate ALM diagnostic id {record_id!r}")
        diagnostics[record_id] = item
    return diagnostics


def _parse_record(path: Path, line_number: int, line: str) -> dict[str, Any]:
    try:
        value = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}:{line_number}: record must be an object")
    return value


def _temporary_path(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(descriptor)
    return Path(name)


def _open_text(path: Path) -> TextIO:
    return path.open("w", encoding="utf-8", newline="\n")


def _write_json_line(handle: TextIO, value: Mapping[str, Any]) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    handle.write("\n")


def _write_text(path: Path, content: str) -> None:
    with _open_text(path) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_path(path: Path) -> None:
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def _output_metadata(
    temporary: Path,
    target: Path,
    *,
    records: int,
) -> dict[str, Any]:
    return {
        "path": str(target),
        "records": records,
        "bytes": temporary.stat().st_size,
        "sha256": sha256_file(temporary),
    }
