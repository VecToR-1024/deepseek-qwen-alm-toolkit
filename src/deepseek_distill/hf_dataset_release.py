"""Build a minimal, auditable Hugging Face release from frozen ALM traces."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .offline_teacher import OfflineTeacherTraceProvider
from .records import NORMALIZED_SCHEMA_VERSION


HF_RELEASE_SCHEMA_VERSION = "offline_alm.hf_release.v1"
ACTUAL_ONLY_RELEASE_PROFILE = "actual_only"
STRICT_TOP20_RELEASE_PROFILE = "strict_top20"
RELEASE_PROFILES = (
    ACTUAL_ONLY_RELEASE_PROFILE,
    STRICT_TOP20_RELEASE_PROFILE,
)

_SOURCE_FIELDS = (
    "dataset",
    "config",
    "split",
    "original_id",
    "revision",
    "license",
    "provenance",
    "mirror",
    "original_name",
    "original_source",
)
_TASK_FIELDS = (
    "schema_version",
    "id",
    "problem_text",
    "interface_type",
    "required_interface",
    "function_name",
    "function_signature",
    "problem_id",
)
_GENERATION_FIELDS = (
    "temperature",
    "top_p",
    "max_tokens",
    "logprobs",
    "top_logprobs",
    "frequency_penalty",
    "presence_penalty",
    "seed",
    "stop",
    "thinking",
)
_USAGE_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "prompt_cache_hit_tokens",
    "prompt_cache_miss_tokens",
)
_SAMPLING_FIELDS = (
    "problem_id",
    "attempt_id",
    "attempt_number",
    "selection_index",
    "selection",
)

_CREDENTIAL_PATTERN = re.compile(
    r"(?i)(?:\bsk[-_][a-z0-9_-]{12,}|\bhf_[a-z0-9]{12,}|bearer\s+[a-z0-9._~+/-]{12,})"
)
_FORBIDDEN_RELEASE_KEYS = {
    "api_key",
    "api_response_id",
    "artifact",
    "authorization",
    "base_url",
    "coding_verification",
    "password",
    "provider",
    "raw_api_json",
    "reference_solution",
    "refresh_token",
    "secret",
    "system_fingerprint",
    "tests",
    "token",
    "top_logprobs",
    "top_probability_mass",
}
_LOCAL_PATH_PATTERNS = (
    re.compile(r"(?i)[a-z]:\\users\\[^\s\"']+"),
    re.compile(r"(?i)(?<![a-z0-9/])/(?:root|home)/[^\s\"']+"),
)


def project_record(
    record: Mapping[str, Any],
    *,
    trace_profile: str = ACTUAL_ONLY_RELEASE_PROFILE,
) -> dict[str, Any]:
    """Whitelist one normalized record into an audited trace release contract."""

    _validate_trace_profile(trace_profile)
    if not isinstance(record, Mapping):
        raise ValueError("training record must be an object")
    if record.get("schema_version") != NORMALIZED_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {NORMALIZED_SCHEMA_VERSION!r}"
        )
    record_id = _required_string(record.get("id"), "record id")
    response_text = record.get("response_text")
    if not isinstance(response_text, str):
        raise ValueError(f"{record_id}: response_text must be a string")

    request = _required_mapping(record.get("request"), f"{record_id}: request")
    generation_config = request.get("generation_config")
    if trace_profile == STRICT_TOP20_RELEASE_PROFILE:
        generation_config = _required_mapping(
            generation_config, f"{record_id}: request.generation_config"
        )
        if generation_config.get("top_logprobs") != 20:
            raise ValueError(
                f"{record_id}: strict_top20 requires generation_config.top_logprobs=20"
            )
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{record_id}: request.messages must be a non-empty list")
    projected_messages: list[dict[str, str]] = []
    local_path_replacements = 0
    for position, message in enumerate(messages):
        message = _required_mapping(
            message, f"{record_id}: request.messages[{position}]"
        )
        sanitized_content, replacements = _redact_local_paths(
            _required_string(
                message.get("content"),
                f"{record_id}: request.messages[{position}].content",
            )
        )
        projected_messages.append(
            {
                "role": _required_string(
                    message.get("role"),
                    f"{record_id}: request.messages[{position}].role",
                ),
                "content": sanitized_content,
            }
        )
        local_path_replacements += replacements

    content_tokens = record.get("content_tokens")
    if not isinstance(content_tokens, list) or not content_tokens:
        raise ValueError(f"{record_id}: content_tokens must be a non-empty list")
    projected_tokens: list[dict[str, Any]] = []
    for position, row in enumerate(content_tokens):
        row = _required_mapping(row, f"{record_id}: content_tokens[{position}]")
        byte_values = row.get("bytes")
        if (
            not isinstance(byte_values, list)
            or not byte_values
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= 255
                for value in byte_values
            )
        ):
            raise ValueError(
                f"{record_id}: content_tokens[{position}].bytes is invalid"
            )
        logprob = row.get("logprob")
        if (
            isinstance(logprob, bool)
            or not isinstance(logprob, (int, float))
            or not math.isfinite(float(logprob))
            or float(logprob) > 1e-7
        ):
            raise ValueError(
                f"{record_id}: content_tokens[{position}].logprob is invalid"
            )
        projected_token: dict[str, Any] = {
            "bytes": list(byte_values),
            "logprob": float(logprob),
        }
        if trace_profile == STRICT_TOP20_RELEASE_PROFILE:
            candidates = _project_strict_top20_candidates(
                row,
                record_id=record_id,
                position=position,
            )
            projected_token["top_logprobs"] = candidates
            projected_token["top_probability_mass"] = min(
                1.0,
                math.fsum(math.exp(candidate["logprob"]) for candidate in candidates),
            )
        projected_tokens.append(projected_token)

    task = _required_mapping(record.get("task"), f"{record_id}: task")
    task_source = _required_mapping(
        task.get("source"), f"{record_id}: task.source"
    )
    projected_source = _copy_fields(task_source, _SOURCE_FIELDS)
    _required_string(projected_source.get("dataset"), f"{record_id}: source dataset")
    projected_task = _copy_fields(task, _TASK_FIELDS)
    for key, value in tuple(projected_task.items()):
        if isinstance(value, str):
            projected_task[key], replacements = _redact_local_paths(value)
            local_path_replacements += replacements
    projected_task["source"] = projected_source

    projected_request: dict[str, Any] = {"messages": projected_messages}
    if isinstance(request.get("model"), str) and request["model"]:
        projected_request["model"] = request["model"]
    if isinstance(generation_config, Mapping):
        projected_request["generation_config"] = _copy_fields(
            generation_config, _GENERATION_FIELDS
        )
    if isinstance(request.get("prompt_contract"), str):
        projected_request["prompt_contract"] = request["prompt_contract"]

    projected: dict[str, Any] = {
        "schema_version": NORMALIZED_SCHEMA_VERSION,
        "id": record_id,
        "release_profile": trace_profile,
        "request": projected_request,
        "response_text": response_text,
        "content_tokens": projected_tokens,
        "task": projected_task,
    }
    for key in ("teacher_model", "finish_reason"):
        if isinstance(record.get(key), str):
            projected[key] = record[key]
    usage = record.get("usage")
    if isinstance(usage, Mapping):
        projected["usage"] = _copy_fields(usage, _USAGE_FIELDS)
    sampling = record.get("sampling")
    if isinstance(sampling, Mapping):
        projected["sampling"] = _copy_fields(sampling, _SAMPLING_FIELDS)
    projected["validation"] = {"content_bytes_match": True}
    projected["release_redactions"] = {
        "local_path_replacements": local_path_replacements
    }

    _scan_safe_strings(projected)
    try:
        OfflineTeacherTraceProvider().get_trace(projected)
    except ValueError as error:
        raise ValueError(f"{record_id}: projected ALM trace is invalid: {error}") from error
    return projected


def package_hf_dataset(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    config_name: str,
    repo_id: str,
    records_per_shard: int = 250,
    expected_records: int | None = None,
    expected_sha256: str | None = None,
    trace_profile: str = ACTUAL_ONLY_RELEASE_PROFILE,
) -> dict[str, Any]:
    """Project and package a frozen JSONL without overwriting any output."""

    _validate_trace_profile(trace_profile)
    source_path = Path(input_path)
    destination = Path(output_dir)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if destination.exists():
        if destination.is_dir() and any(destination.iterdir()):
            raise FileExistsError(f"output directory is not empty: {destination}")
        raise FileExistsError(f"output path already exists: {destination}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", config_name):
        raise ValueError("config_name may contain only letters, digits, dot, dash, underscore")
    if not re.fullmatch(r"[^/\s]+/[^/\s]+", repo_id):
        raise ValueError("repo_id must be in namespace/name form")
    if (
        isinstance(records_per_shard, bool)
        or not isinstance(records_per_shard, int)
        or records_per_shard <= 0
    ):
        raise ValueError("records_per_shard must be a positive integer")

    input_sha256 = _sha256(source_path)
    if expected_sha256 is not None and input_sha256 != expected_sha256.lower():
        raise ValueError(
            f"input SHA256 mismatch: expected {expected_sha256.lower()}, got {input_sha256}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.", dir=destination.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        shard_dir = temporary / "data" / config_name
        shard_dir.mkdir(parents=True)
        licenses: dict[str, set[str]] = defaultdict(set)
        source_counts: Counter[str] = Counter()
        redactions: Counter[str] = Counter()
        trace_counts: Counter[str] = Counter()
        temporary_shards: list[Path] = []
        record_count = 0
        shard_handle: io.TextIOWrapper | None = None

        def close_shard() -> None:
            nonlocal shard_handle
            if shard_handle is not None:
                shard_handle.close()
                shard_handle = None

        try:
            with source_path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        original = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"input line {line_number} is not valid JSON"
                        ) from error
                    projected = project_record(
                        original,
                        trace_profile=trace_profile,
                    )
                    if record_count % records_per_shard == 0:
                        close_shard()
                        part_path = shard_dir / f"part-{len(temporary_shards):05d}.jsonl.gz"
                        temporary_shards.append(part_path)
                        shard_handle = _open_deterministic_gzip_text(part_path)
                    assert shard_handle is not None
                    shard_handle.write(
                        json.dumps(
                            projected,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    record_count += 1
                    _accumulate_trace_counts(
                        projected,
                        trace_profile=trace_profile,
                        counts=trace_counts,
                    )
                    source = projected["task"]["source"]
                    dataset = str(source["dataset"])
                    source_counts[dataset] += 1
                    licenses[dataset].add(str(source.get("license", "unspecified")))
                    release_redactions = projected.get("release_redactions")
                    if isinstance(release_redactions, Mapping):
                        redactions["prompt_local_path_replacements"] += int(
                            release_redactions.get("local_path_replacements", 0)
                        )
                    if isinstance(original.get("task"), Mapping) and "tests" in original["task"]:
                        redactions["official_tests"] += 1
                    if trace_profile == ACTUAL_ONLY_RELEASE_PROFILE and any(
                        isinstance(row, Mapping) and row.get("top_logprobs")
                        for row in original.get("content_tokens", [])
                    ):
                        redactions["top_logprob_distributions"] += 1
                    for key, label in (
                        ("api_response_id", "api_response_ids"),
                        ("system_fingerprint", "system_fingerprints"),
                        ("provider", "provider_details"),
                        ("coding_verification", "verifier_details"),
                    ):
                        if key in original:
                            redactions[label] += 1
        finally:
            close_shard()

        if record_count == 0:
            raise ValueError("input dataset contains no records")
        if expected_records is not None and record_count != expected_records:
            raise ValueError(
                f"record count mismatch: expected {expected_records}, got {record_count}"
            )

        shard_metadata: list[dict[str, Any]] = []
        shard_total = len(temporary_shards)
        for index, temporary_shard in enumerate(temporary_shards):
            final_name = f"train-{index:05d}-of-{shard_total:05d}.jsonl.gz"
            final_path = temporary_shard.with_name(final_name)
            temporary_shard.replace(final_path)
            shard_metadata.append(
                {
                    "path": final_path.relative_to(temporary).as_posix(),
                    "records": min(
                        records_per_shard,
                        record_count - index * records_per_shard,
                    ),
                    "bytes": final_path.stat().st_size,
                    "sha256": _sha256(final_path),
                }
            )

        manifest = {
            "schema_version": HF_RELEASE_SCHEMA_VERSION,
            "repo_id": repo_id,
            "config_name": config_name,
            "projection": {
                "policy": "allowlist_only",
                "alm_semantics": "exact_actual_token_bytes_and_logprobs",
                "trace_profile": trace_profile,
                "strict_top20_candidates_retained": (
                    trace_profile == STRICT_TOP20_RELEASE_PROFILE
                ),
                "tail_bucket_mass_reconstructable": (
                    trace_profile == STRICT_TOP20_RELEASE_PROFILE
                ),
                "source_record_schema": NORMALIZED_SCHEMA_VERSION,
                "published_record_schema": NORMALIZED_SCHEMA_VERSION,
            },
            "input": {
                "filename": source_path.name,
                "bytes": source_path.stat().st_size,
                "sha256": input_sha256,
            },
            "counts": {"records": record_count, "shards": shard_total},
            "trace_counts": {
                "actual_token_positions": trace_counts["actual_token_positions"],
                "top_logprob_candidates": trace_counts["top_logprob_candidates"],
                "positions_with_exact_top20": trace_counts[
                    "positions_with_exact_top20"
                ],
            },
            "source_counts": dict(sorted(source_counts.items())),
            "source_licenses": {
                dataset: sorted(values) for dataset, values in sorted(licenses.items())
            },
            "redactions": {
                key: redactions.get(key, 0)
                for key in (
                    "official_tests",
                    "prompt_local_path_replacements",
                    "top_logprob_distributions",
                    "api_response_ids",
                    "system_fingerprints",
                    "provider_details",
                    "verifier_details",
                )
            },
            "outputs": {"shards": shard_metadata},
        }
        manifest_path = temporary / "release_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "README.md").write_text(
            _render_dataset_card(manifest), encoding="utf-8"
        )
        temporary.replace(destination)
    return manifest


def upload_hf_package(
    *,
    package_dir: str | Path,
    repo_id: str,
    confirmed_manifest_sha256: str,
    private: bool = True,
    commit_message: str = "Upload audited offline ALM training traces",
    api: Any | None = None,
) -> dict[str, Any]:
    """Create a dataset repository, upload one package, and verify its file set.

    Authentication is delegated to ``huggingface_hub``. In unattended use, set
    ``HF_TOKEN`` in the process environment; never pass or persist a token here.
    """

    package_path = Path(package_dir)
    manifest_path = package_path / "release_manifest.json"
    card_path = package_path / "README.md"
    if not package_path.is_dir() or not manifest_path.is_file() or not card_path.is_file():
        raise ValueError(
            "package_dir must contain README.md and release_manifest.json"
        )
    manifest_sha256 = _sha256(manifest_path)
    if confirmed_manifest_sha256.lower() != manifest_sha256:
        raise ValueError(
            "human-confirmed manifest SHA256 does not match the package: "
            f"expected {confirmed_manifest_sha256.lower()}, got {manifest_sha256}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("release_manifest.json is not valid JSON") from error
    if not isinstance(manifest, Mapping) or manifest.get(
        "schema_version"
    ) != HF_RELEASE_SCHEMA_VERSION:
        raise ValueError("release_manifest.json has an unsupported schema")
    manifest_repo_id = manifest.get("repo_id")
    if manifest_repo_id != repo_id:
        raise ValueError(
            f"manifest repo_id is {manifest_repo_id!r}, not requested {repo_id!r}"
        )
    local_files = sorted(
        path.relative_to(package_path).as_posix()
        for path in package_path.rglob("*")
        if path.is_file()
    )
    expected_files = {"README.md", "release_manifest.json"}
    expected_files.update(
        str(row["path"]) for row in manifest["outputs"]["shards"]
    )
    if set(local_files) != expected_files:
        missing = sorted(expected_files - set(local_files))
        unexpected = sorted(set(local_files) - expected_files)
        raise ValueError(
            f"package file set differs from manifest; missing={missing}, "
            f"unexpected={unexpected}"
        )

    audit = audit_hf_package(package_path)

    if api is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as error:
            raise RuntimeError(
                "install the 'release' extra to upload: "
                "pip install -e '.[release]'"
            ) from error
        api = HfApi()
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )
    commit_url = api.upload_folder(
        folder_path=str(package_path),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
        ignore_patterns=["**/.git/**", "**/__pycache__/**"],
    )
    remote_files = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
    missing_remote = sorted(expected_files - remote_files)
    if missing_remote:
        raise RuntimeError(f"remote upload verification failed; missing={missing_remote}")
    return {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "visibility": "private" if private else "public",
        "commit_url": str(commit_url),
        "manifest_sha256": manifest_sha256,
        "audited_records": audit["counts"]["records"],
        "verified_files": len(expected_files),
    }


def audit_hf_package(package_dir: str | Path) -> dict[str, Any]:
    """Independently revalidate a release before any Hub mutation."""

    package_path = Path(package_dir)
    manifest_path = package_path / "release_manifest.json"
    card_path = package_path / "README.md"
    if not package_path.is_dir() or not manifest_path.is_file() or not card_path.is_file():
        raise ValueError(
            "package_dir must contain README.md and release_manifest.json"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("release_manifest.json is not valid JSON") from error
    if not isinstance(manifest, Mapping) or manifest.get(
        "schema_version"
    ) != HF_RELEASE_SCHEMA_VERSION:
        raise ValueError("release_manifest.json has an unsupported schema")
    projection = manifest.get("projection")
    trace_profile = (
        projection.get("trace_profile")
        if isinstance(projection, Mapping)
        else ACTUAL_ONLY_RELEASE_PROFILE
    )
    if trace_profile is None:
        trace_profile = ACTUAL_ONLY_RELEASE_PROFILE
    _validate_trace_profile(trace_profile)
    outputs = manifest.get("outputs")
    shard_rows = outputs.get("shards") if isinstance(outputs, Mapping) else None
    if not isinstance(shard_rows, list) or not shard_rows:
        raise ValueError("release manifest contains no shards")

    expected_files = {"README.md", "release_manifest.json"}
    expected_files.update(str(row["path"]) for row in shard_rows)
    actual_files = {
        path.relative_to(package_path).as_posix()
        for path in package_path.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError(
            "package file set differs from manifest; "
            f"missing={sorted(expected_files - actual_files)}, "
            f"unexpected={sorted(actual_files - expected_files)}"
        )
    _scan_safe_strings(card_path.read_text(encoding="utf-8"), "README.md")
    _scan_safe_strings(manifest, "release_manifest")

    provider = OfflineTeacherTraceProvider()
    seen_ids: set[str] = set()
    duplicate_ids = 0
    record_count = 0
    trace_records = 0
    trace_counts: Counter[str] = Counter()
    for shard_index, shard_row in enumerate(shard_rows):
        if not isinstance(shard_row, Mapping):
            raise ValueError(f"shard metadata {shard_index} must be an object")
        relative_path = Path(str(shard_row.get("path", "")))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"shard path is not safe: {relative_path}")
        shard_path = package_path / relative_path
        if shard_path.stat().st_size != shard_row.get("bytes"):
            raise ValueError(f"shard byte size mismatch: {relative_path.as_posix()}")
        shard_sha256 = _sha256(shard_path)
        if shard_sha256 != shard_row.get("sha256"):
            raise ValueError(
                f"shard SHA256 mismatch: {relative_path.as_posix()}"
            )
        shard_records = 0
        with gzip.open(shard_path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"{relative_path.as_posix()}:{line_number} is invalid JSON"
                    ) from error
                if not isinstance(record, Mapping):
                    raise ValueError(
                        f"{relative_path.as_posix()}:{line_number} must be an object"
                    )
                _scan_forbidden_release_keys(
                    record,
                    trace_profile=trace_profile,
                )
                _scan_safe_strings(
                    record, f"{relative_path.as_posix()}:{line_number}"
                )
                trace = provider.get_trace(record)
                if record.get("release_profile", trace_profile) != trace_profile:
                    raise ValueError(
                        f"{relative_path.as_posix()}:{line_number} release_profile "
                        "does not match the manifest"
                    )
                _accumulate_trace_counts(
                    record,
                    trace_profile=trace_profile,
                    counts=trace_counts,
                )
                trace_records += 1
                if trace.record_id in seen_ids:
                    duplicate_ids += 1
                seen_ids.add(trace.record_id)
                record_count += 1
                shard_records += 1
        if shard_records != shard_row.get("records"):
            raise ValueError(
                f"shard record count mismatch: {relative_path.as_posix()}"
            )

    expected_counts = manifest.get("counts")
    expected_records = (
        expected_counts.get("records") if isinstance(expected_counts, Mapping) else None
    )
    if record_count != expected_records:
        raise ValueError(
            f"release record count mismatch: expected {expected_records}, got {record_count}"
        )
    if duplicate_ids:
        raise ValueError(f"release contains {duplicate_ids} duplicate record IDs")
    audited_trace_counts = {
        "actual_token_positions": trace_counts["actual_token_positions"],
        "top_logprob_candidates": trace_counts["top_logprob_candidates"],
        "positions_with_exact_top20": trace_counts[
            "positions_with_exact_top20"
        ],
    }
    manifest_trace_counts = manifest.get("trace_counts")
    if isinstance(manifest_trace_counts, Mapping) and dict(
        manifest_trace_counts
    ) != audited_trace_counts:
        raise ValueError(
            "release trace counts differ from manifest: "
            f"expected {dict(manifest_trace_counts)}, got {audited_trace_counts}"
        )
    return {
        "schema_version": "offline_alm.hf_release_audit.v1",
        "package_manifest_sha256": _sha256(manifest_path),
        "trace_profile": trace_profile,
        "counts": {
            "records": record_count,
            "unique_ids": len(seen_ids),
            "duplicate_ids": duplicate_ids,
            "shards": len(shard_rows),
        },
        "trace_reconstruction_records": trace_records,
        "trace_counts": audited_trace_counts,
        "sensitive_scan": {
            "credential_like_values": 0,
            "forbidden_keys": 0,
            "local_paths": 0,
        },
    }


def _open_deterministic_gzip_text(path: Path) -> io.TextIOWrapper:
    binary = path.open("wb")
    compressed = gzip.GzipFile(
        filename="", mode="wb", compresslevel=9, fileobj=binary, mtime=0
    )
    return io.TextIOWrapper(compressed, encoding="utf-8", newline="\n")


def _render_dataset_card(manifest: Mapping[str, Any]) -> str:
    config_name = str(manifest["config_name"])
    repo_id = str(manifest["repo_id"])
    records = int(manifest["counts"]["records"])
    source_counts = manifest["source_counts"]
    source_licenses = manifest["source_licenses"]
    projection = manifest.get("projection")
    trace_profile = (
        projection.get("trace_profile")
        if isinstance(projection, Mapping)
        else ACTUAL_ONLY_RELEASE_PROFILE
    )
    source_rows = "\n".join(
        f"| {dataset} | {source_counts[dataset]} | "
        f"{', '.join(source_licenses[dataset])} |"
        for dataset in source_counts
    )
    if trace_profile == STRICT_TOP20_RELEASE_PROFILE:
        pretty_name = "DeepSeek to Qwen Strict Top-20 Traces"
        heading = "DeepSeek to Qwen Strict Top-20 Traces"
        trace_description = (
            "the actual generated-token trace plus exactly 20 alternative-token "
            "byte/logprob candidates at every generated position"
        )
        token_fields = (
            "- `content_tokens[].bytes` and `content_tokens[].logprob`: the actual "
            "teacher trajectory.\n"
            "- `content_tokens[].top_logprobs`: exactly 20 candidate UTF-8 byte "
            "sequences, token strings, and log probabilities per position.\n"
            "- `content_tokens[].top_probability_mass`: probability mass recomputed "
            "from the retained candidates; the complement defines the tail bucket."
        )
        exclusion_text = (
            "Official tests are intentionally excluded. Reference solutions, verifier "
            "output, local filesystem paths, provider response identifiers, system "
            "fingerprints, and actual-token display strings are also excluded. The "
            "strict top-20 trace supports the experimental top-20 + tail bucket "
            "baseline; it does not expose the provider's full vocabulary distribution."
        )
    else:
        pretty_name = "DeepSeek to Qwen Offline ALM Traces"
        heading = "DeepSeek to Qwen Offline ALM Traces"
        trace_description = (
            "the actual generated-token UTF-8 bytes and log probabilities needed for "
            "offline Approximate Likelihood Matching (ALM)"
        )
        token_fields = (
            "- `content_tokens[].bytes` and `content_tokens[].logprob`: the exact "
            "teacher trajectory consumed by the ALM implementation."
        )
        exclusion_text = (
            "Official tests are intentionally excluded. Reference solutions, verifier "
            "output, local filesystem paths, provider response identifiers, system "
            "fingerprints, and top-k alternative-token distributions are also excluded. "
            "The published trajectory still reconstructs "
            "`response_text.encode(\"utf-8\")` exactly for every record."
        )
    return f"""---
pretty_name: {pretty_name}
license: other
task_categories:
- text-generation
configs:
- config_name: {config_name}
  data_files:
  - split: train
    path: data/{config_name}/*.jsonl.gz
---

# {heading}

This private-first release contains {records} verified Python coding solutions with
{trace_description}.

```python
from datasets import load_dataset

train = load_dataset({repo_id!r}, {config_name!r}, split="train")
```

## Published fields

- `request.messages`: the teacher prompts, with any local user paths replaced
  by `<LOCAL_PATH>`.
- `response_text`: the unmodified accepted teacher completion.
{token_fields}
- `task.source`: provenance, pinned revision, split, and per-source license labels.
- compact generation, sampling, and token-usage metadata where available.

{exclusion_text}

## Sources and licensing

This is a mixed-source research dataset, so no single permissive license is claimed.
Review each upstream dataset's terms and caveats before changing repository
visibility or redistributing the data.

| Source | Records | Recorded license label(s) |
| --- | ---: | --- |
{source_rows}

## Reproducibility

See `release_manifest.json` for the authoritative input SHA256, record counts,
redaction counts, deterministic shard hashes, and source distribution.
"""


def _validate_trace_profile(trace_profile: str) -> None:
    if trace_profile not in RELEASE_PROFILES:
        raise ValueError(
            f"trace_profile must be one of {', '.join(RELEASE_PROFILES)}"
        )


def _project_strict_top20_candidates(
    row: Mapping[str, Any],
    *,
    record_id: str,
    position: int,
) -> list[dict[str, Any]]:
    candidates = row.get("top_logprobs")
    if not isinstance(candidates, list) or len(candidates) != 20:
        actual = len(candidates) if isinstance(candidates, list) else "missing"
        raise ValueError(
            f"{record_id}: content_tokens[{position}].top_logprobs must contain "
            f"exactly 20 candidates, got {actual}"
        )
    projected: list[dict[str, Any]] = []
    for candidate_position, candidate_value in enumerate(candidates):
        candidate = _required_mapping(
            candidate_value,
            f"{record_id}: content_tokens[{position}].top_logprobs"
            f"[{candidate_position}]",
        )
        byte_values = candidate.get("bytes")
        if (
            not isinstance(byte_values, list)
            or not byte_values
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= 255
                for value in byte_values
            )
        ):
            raise ValueError(
                f"{record_id}: content_tokens[{position}].top_logprobs"
                f"[{candidate_position}].bytes is invalid"
            )
        logprob = candidate.get("logprob")
        if (
            isinstance(logprob, bool)
            or not isinstance(logprob, (int, float))
            or not math.isfinite(float(logprob))
            or float(logprob) > 1e-7
        ):
            raise ValueError(
                f"{record_id}: content_tokens[{position}].top_logprobs"
                f"[{candidate_position}].logprob is invalid"
            )
        token = candidate.get("token")
        if not isinstance(token, str):
            raise ValueError(
                f"{record_id}: content_tokens[{position}].top_logprobs"
                f"[{candidate_position}].token must be a string"
            )
        projected.append(
            {
                "token": token,
                "bytes": list(byte_values),
                "logprob": float(logprob),
            }
        )
    return projected


def _accumulate_trace_counts(
    record: Mapping[str, Any],
    *,
    trace_profile: str,
    counts: Counter[str],
) -> None:
    content_tokens = record.get("content_tokens")
    if not isinstance(content_tokens, list) or not content_tokens:
        raise ValueError("published content_tokens must be a non-empty list")
    for position, row_value in enumerate(content_tokens):
        row = _required_mapping(row_value, f"content_tokens[{position}]")
        counts["actual_token_positions"] += 1
        candidates = row.get("top_logprobs")
        if trace_profile == STRICT_TOP20_RELEASE_PROFILE:
            projected_candidates = _project_strict_top20_candidates(
                row,
                record_id=str(record.get("id", "record")),
                position=position,
            )
            counts["top_logprob_candidates"] += len(projected_candidates)
            counts["positions_with_exact_top20"] += 1
            expected_mass = min(
                1.0,
                math.fsum(
                    math.exp(candidate["logprob"])
                    for candidate in projected_candidates
                ),
            )
            stored_mass = row.get("top_probability_mass")
            if (
                isinstance(stored_mass, bool)
                or not isinstance(stored_mass, (int, float))
                or not math.isfinite(float(stored_mass))
                or not math.isclose(
                    float(stored_mass), expected_mass, rel_tol=1e-12, abs_tol=1e-15
                )
            ):
                raise ValueError(
                    f"{record.get('id', 'record')}: content_tokens[{position}] "
                    "top_probability_mass does not match candidate logprobs"
                )
        elif candidates is not None:
            raise ValueError(
                f"{record.get('id', 'record')}: actual_only release contains "
                f"top_logprobs at content_tokens[{position}]"
            )


def _required_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _copy_fields(source: Mapping[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    return {name: source[name] for name in names if name in source}


def _scan_safe_strings(value: Any, path: str = "record") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _scan_safe_strings(child, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _scan_safe_strings(child, f"{path}[{index}]")
        return
    if not isinstance(value, str):
        return
    if _CREDENTIAL_PATTERN.search(value):
        raise ValueError(f"{path} contains a credential-like value")
    if any(pattern.search(value) for pattern in _LOCAL_PATH_PATTERNS):
        raise ValueError(f"{path} contains a local filesystem path")


def _scan_forbidden_release_keys(
    value: Any,
    path: str = "record",
    *,
    trace_profile: str = ACTUAL_ONLY_RELEASE_PROFILE,
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            generation_parameter = (
                normalized == "top_logprobs"
                and path == "record.request.generation_config"
            )
            strict_trace_field = (
                trace_profile == STRICT_TOP20_RELEASE_PROFILE
                and normalized in {"top_logprobs", "top_probability_mass"}
                and re.fullmatch(r"record\.content_tokens\[\d+\]", path)
                is not None
            )
            strict_candidate_token = (
                trace_profile == STRICT_TOP20_RELEASE_PROFILE
                and normalized == "token"
                and re.fullmatch(
                    r"record\.content_tokens\[\d+\]\.top_logprobs\[\d+\]",
                    path,
                )
                is not None
            )
            if (
                normalized in _FORBIDDEN_RELEASE_KEYS
                and not generation_parameter
                and not strict_trace_field
                and not strict_candidate_token
            ):
                raise ValueError(f"{path}.{key} is a forbidden release field")
            _scan_forbidden_release_keys(
                child,
                f"{path}.{key}",
                trace_profile=trace_profile,
            )
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _scan_forbidden_release_keys(
                child,
                f"{path}[{index}]",
                trace_profile=trace_profile,
            )


def _redact_local_paths(value: str) -> tuple[str, int]:
    redacted = value
    replacements = 0
    for pattern in _LOCAL_PATH_PATTERNS:
        redacted, count = pattern.subn("<LOCAL_PATH>", redacted)
        replacements += count
    return redacted, replacements


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
