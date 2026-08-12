"""Final clean-candidate decisions for verified multi-source coding traces."""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .clean_eligibility import CleanEligibilityPolicy, evaluate_clean_eligibility


AUDIT_SCHEMA_VERSION = "offline_alm.multisource_clean_audit.v1"
EXCLUSION_SCHEMA_VERSION = "offline_alm.clean_exclusion.v1"


def eos_supervision_by_record(
    record_ids: Sequence[str],
    training_contract: Mapping[str, Any],
) -> dict[str, bool]:
    """Reconstruct and validate per-record EOS supervision from the contract audit."""

    if len(record_ids) != len(set(record_ids)):
        raise ValueError("record IDs must be unique")
    record_id_set = set(record_ids)
    records = _integer(training_contract.get("records"))
    ends = training_contract.get("end_token_supervision")
    if records != len(record_ids) or not isinstance(ends, Mapping):
        raise ValueError("training contract record count does not match accepted records")

    missing = _string_set(ends.get("missing_eos_record_ids"), "missing EOS")
    ignored = _string_set(ends.get("ignored_eos_record_ids"), "ignored EOS")
    unknown = (missing | ignored) - record_id_set
    if unknown:
        raise ValueError(f"EOS contract contains unknown record IDs: {sorted(unknown)!r}")
    if missing & ignored:
        raise ValueError("an EOS record cannot be both missing and ignored")

    eos_present = _integer(ends.get("eos_present_records"))
    eos_supervised = _integer(ends.get("eos_supervised_records"))
    if eos_present != len(record_ids) - len(missing):
        raise ValueError("EOS-present aggregate does not reconcile with record IDs")
    unsupervised = missing | ignored
    if eos_supervised != len(record_ids) - len(unsupervised):
        raise ValueError("EOS-supervised aggregate does not reconcile with record IDs")
    return {record_id: record_id not in unsupervised for record_id in record_ids}


def build_multisource_clean_audit(
    *,
    records: Sequence[Mapping[str, Any]],
    alm: Mapping[str, Any],
    training_contract: Mapping[str, Any],
    policy: CleanEligibilityPolicy | None = None,
) -> dict[str, Any]:
    """Apply the complete trace, format, ALM, length, and EOS eligibility gate."""

    selected_policy = policy or CleanEligibilityPolicy()
    record_ids = [_record_id(record) for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("accepted records contain duplicate IDs")
    eos_by_id = eos_supervision_by_record(record_ids, training_contract)
    diagnostics = _diagnostics_by_id(alm)

    decisions: list[dict[str, Any]] = []
    retained: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    source_counts: dict[str, Counter[str]] = {}
    for record in records:
        record_id = _record_id(record)
        decision = evaluate_clean_eligibility(
            record,
            alm_diagnostic=diagnostics.get(record_id),
            eos_supervised=eos_by_id[record_id],
            policy=selected_policy,
        )
        decisions.append(decision)
        source = str(decision["source"])
        per_source = source_counts.setdefault(source, Counter())
        per_source["official_test_passed"] += 1
        if decision["eligible"]:
            per_source["clean_eligible"] += 1
            retained.append(copy.deepcopy(dict(record)))
        else:
            per_source["clean_excluded"] += 1
            reason_counts.update(decision["reasons"])
            excluded.append(
                {
                    "schema_version": EXCLUSION_SCHEMA_VERSION,
                    "id": record_id,
                    "source": source,
                    "reasons": list(decision["reasons"]),
                }
            )

    report = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "counts": {
            "official_test_passed": len(records),
            "clean_eligible": len(retained),
            "clean_excluded": len(excluded),
        },
        "source_counts": {
            source: {
                "official_test_passed": counts["official_test_passed"],
                "clean_eligible": counts["clean_eligible"],
                "clean_excluded": counts["clean_excluded"],
            }
            for source, counts in sorted(source_counts.items())
        },
        "reason_counts": dict(sorted(reason_counts.items())),
        "policy": {
            "max_comment_line_ratio": selected_policy.max_comment_line_ratio,
            "max_sequence_length": selected_policy.max_sequence_length,
            "raw_trace_mutation_allowed": False,
        },
        "eos_supervised_records": sum(eos_by_id.values()),
        "alm_preprocessing_errors": len(alm.get("preprocessing_errors") or []),
        "zero_alm_chunk_records": len(
            alm.get("examples_with_zero_valid_chunks") or []
        ),
        "records_exceeding_max_length": len(
            alm.get("records_exceeding_max_length") or []
        ),
    }
    return {
        "report": report,
        "decisions": decisions,
        "retained_records": retained,
        "excluded_records": excluded,
    }


def _diagnostics_by_id(alm: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    examples = alm.get("examples")
    if not isinstance(examples, list):
        raise ValueError("ALM diagnostics must contain an examples list")
    result: dict[str, Mapping[str, Any]] = {}
    for index, example in enumerate(examples):
        if not isinstance(example, Mapping):
            raise ValueError(f"ALM examples[{index}] must be an object")
        record_id = _record_id(example)
        if record_id in result:
            raise ValueError(f"duplicate ALM diagnostic ID {record_id!r}")
        result[record_id] = example
    return result


def _record_id(record: Mapping[str, Any]) -> str:
    record_id = record.get("id")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("record has no non-empty string id")
    return record_id


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _string_set(value: Any, label: str) -> set[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"{label} IDs must be a list of non-empty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} IDs contain duplicates")
    return set(value)
