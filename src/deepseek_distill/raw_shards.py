"""Seal terminal raw JSONL queues into verified Zstandard archives."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .durable_io import replace_file
from .records import RAW_SCHEMA_VERSION
from .rejection_sampling import publish_json_once


RAW_SHARD_SEAL_SCHEMA_VERSION = "deepseek.raw.shard.seal.v1"
RAW_SHARD_ARCHIVE_SCHEMA_VERSION = "deepseek.raw.shard.archive.v1"


@dataclass(frozen=True, slots=True)
class RawShardSealSummary:
    status: str
    records: int
    successful_records: int
    api_error_records: int
    source_bytes: int
    source_sha256: str
    archive_bytes: int
    archive_sha256: str
    source_removed: bool


@dataclass(frozen=True, slots=True)
class _RawScan:
    ids: tuple[str, ...]
    successful_ids: frozenset[str]
    api_error_ids: frozenset[str]
    byte_count: int
    sha256: str


@dataclass(frozen=True, slots=True)
class _ArchiveScan:
    archive_bytes: int
    archive_sha256: str
    decompressed_bytes: int
    decompressed_sha256: str
    decompressed_records: int


def seal_raw_shard(
    *,
    raw_path: Path,
    normalized_path: Path,
    normalization_errors_path: Path,
    verifier_path: Path,
    state_path: Path,
    archive_path: Path,
    seal_manifest_path: Path,
    archive_manifest_path: Path,
    expected_records: int | None = None,
    compression_level: int = 6,
    remove_source: bool = False,
) -> RawShardSealSummary:
    """Compress one fully drained durable raw queue and verify its round trip."""

    raw_path = Path(raw_path)
    archive_path = Path(archive_path)
    seal_manifest_path = Path(seal_manifest_path)
    archive_manifest_path = Path(archive_manifest_path)
    if expected_records is not None and (
        isinstance(expected_records, bool)
        or not isinstance(expected_records, int)
        or expected_records <= 0
    ):
        raise ValueError("expected_records must be a positive integer or null")
    if isinstance(compression_level, bool) or not isinstance(compression_level, int):
        raise ValueError("compression_level must be an integer")

    if not raw_path.is_file():
        return _resume_without_source(
            archive_path=archive_path,
            seal_manifest_path=seal_manifest_path,
            archive_manifest_path=archive_manifest_path,
            expected_records=expected_records,
        )

    raw = _scan_raw_jsonl(raw_path)
    if expected_records is not None and len(raw.ids) != expected_records:
        raise ValueError(
            f"raw shard contains {len(raw.ids)} records, expected {expected_records}"
        )
    downstream = _validate_downstream_acknowledgement(
        raw=raw,
        normalized_path=Path(normalized_path),
        normalization_errors_path=Path(normalization_errors_path),
        verifier_path=Path(verifier_path),
        state_path=Path(state_path),
    )
    seal = {
        "schema_version": RAW_SHARD_SEAL_SCHEMA_VERSION,
        "source": {
            "file_name": raw_path.name,
            "records": len(raw.ids),
            "successful_records": len(raw.successful_ids),
            "api_error_records": len(raw.api_error_ids),
            "bytes": raw.byte_count,
            "sha256": raw.sha256,
            "ordered_ids_sha256": hashlib.sha256(
                "\n".join(raw.ids).encode("utf-8")
            ).hexdigest(),
        },
        "downstream_acknowledged": True,
        "downstream": downstream,
        "compression_policy": {
            "algorithm": "zstd",
            "level": compression_level,
        },
    }
    publish_json_once(seal_manifest_path, seal)

    created = False
    if archive_path.exists():
        archive = _verify_archive(archive_path, expected=raw)
    else:
        _compress_atomic(
            raw_path,
            archive_path,
            compression_level=compression_level,
        )
        created = True
        archive = _verify_archive(archive_path, expected=raw)

    archive_manifest = {
        "schema_version": RAW_SHARD_ARCHIVE_SCHEMA_VERSION,
        "seal_manifest": {
            "file_name": seal_manifest_path.name,
            "sha256": _sha256_file(seal_manifest_path),
        },
        "archive": {
            "file_name": archive_path.name,
            "compression": "zstd",
            "compression_level": compression_level,
            "bytes": archive.archive_bytes,
            "sha256": archive.archive_sha256,
        },
        "verification": {
            "decompressed_bytes": archive.decompressed_bytes,
            "decompressed_sha256": archive.decompressed_sha256,
            "decompressed_records": archive.decompressed_records,
        },
    }
    publish_json_once(archive_manifest_path, archive_manifest)

    source_removed = False
    if remove_source:
        raw_path.unlink()
        source_removed = True
    return RawShardSealSummary(
        status="created" if created else "unchanged",
        records=len(raw.ids),
        successful_records=len(raw.successful_ids),
        api_error_records=len(raw.api_error_ids),
        source_bytes=raw.byte_count,
        source_sha256=raw.sha256,
        archive_bytes=archive.archive_bytes,
        archive_sha256=archive.archive_sha256,
        source_removed=source_removed,
    )


def _validate_downstream_acknowledgement(
    *,
    raw: _RawScan,
    normalized_path: Path,
    normalization_errors_path: Path,
    verifier_path: Path,
    state_path: Path,
) -> dict[str, int]:
    state = _read_object(state_path)
    if state.get("phase") != "completed":
        raise ValueError("pipeline state must be completed before sealing a raw shard")
    queues = state.get("queues")
    if not isinstance(queues, Mapping):
        raise ValueError("pipeline state queues must be an object")
    if queues.get("raw_to_normalized_lag") != 0:
        raise ValueError("pipeline has an unacknowledged raw-to-normalized lag")
    if queues.get("normalized_to_verifier_lag") != 0:
        raise ValueError("pipeline has an unacknowledged normalized-to-verifier lag")

    normalized_ids = _read_unique_ids(normalized_path)
    normalization_error_ids = _read_unique_ids(normalization_errors_path)
    verifier_ids = _read_unique_ids(verifier_path)
    normalization_terminal = normalized_ids | normalization_error_ids
    missing_normalization = raw.successful_ids - normalization_terminal
    if missing_normalization:
        raise ValueError(
            "successful raw IDs without normalization terminal results: "
            f"{sorted(missing_normalization)[:3]}"
        )
    unexpected_normalization = normalization_terminal - raw.successful_ids
    if unexpected_normalization:
        raise ValueError("normalization queues contain IDs outside successful raw records")
    missing_verifier = raw.successful_ids - verifier_ids
    if missing_verifier:
        raise ValueError(
            "successful raw IDs without verifier results: "
            f"{sorted(missing_verifier)[:3]}"
        )
    if verifier_ids - raw.successful_ids:
        raise ValueError("verifier queue contains IDs outside successful raw records")

    expected_counts = {
        "raw": len(raw.ids),
        "normalized": len(normalized_ids),
        "normalization_errors": len(normalization_error_ids),
        "verifier": len(verifier_ids),
    }
    for key, expected in expected_counts.items():
        if queues.get(key) != expected:
            raise ValueError(f"pipeline state queue count {key!r} does not match disk")
    return expected_counts


def _scan_raw_jsonl(path: Path) -> _RawScan:
    digest = hashlib.sha256()
    ids: list[str] = []
    successful: set[str] = set()
    api_errors: set[str] = set()
    byte_count = 0
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            digest.update(line)
            byte_count += len(line)
            if not line.endswith(b"\n"):
                raise ValueError(f"{path}:{line_number}: incomplete trailing JSONL record")
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL records are not allowed")
            try:
                value = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number}: expected an object")
            if value.get("schema_version") != RAW_SCHEMA_VERSION:
                raise ValueError(f"{path}:{line_number}: invalid raw schema_version")
            record_id = value.get("id")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"{path}:{line_number}: invalid id")
            if record_id in successful or record_id in api_errors:
                raise ValueError(f"{path}:{line_number}: duplicate id {record_id!r}")
            status = value.get("status")
            if status == "ok":
                successful.add(record_id)
            elif status == "error":
                api_errors.add(record_id)
            else:
                raise ValueError(f"{path}:{line_number}: invalid raw status")
            ids.append(record_id)
    if not ids:
        raise ValueError("raw shard must contain at least one record")
    return _RawScan(
        ids=tuple(ids),
        successful_ids=frozenset(successful),
        api_error_ids=frozenset(api_errors),
        byte_count=byte_count,
        sha256=digest.hexdigest(),
    )


def _compress_atomic(source: Path, archive: Path, *, compression_level: int) -> None:
    zstandard = _load_zstandard()
    archive.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{archive.name}.", suffix=".tmp", dir=archive.parent
    )
    try:
        with source.open("rb") as source_handle, os.fdopen(descriptor, "wb") as output:
            compressor = zstandard.ZstdCompressor(
                level=compression_level,
                write_checksum=True,
                write_content_size=True,
            )
            with compressor.stream_writer(
                output,
                size=source.stat().st_size,
                closefd=False,
            ) as compressed:
                shutil.copyfileobj(source_handle, compressed, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        replace_file(temporary, archive)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _verify_archive(path: Path, *, expected: _RawScan) -> _ArchiveScan:
    zstandard = _load_zstandard()
    digest = hashlib.sha256()
    byte_count = 0
    newline_count = 0
    last_byte = b""
    try:
        with path.open("rb") as source:
            with zstandard.ZstdDecompressor().stream_reader(
                source,
                closefd=False,
            ) as reader:
                while True:
                    chunk = reader.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    byte_count += len(chunk)
                    newline_count += chunk.count(b"\n")
                    last_byte = chunk[-1:]
    except Exception as error:  # noqa: BLE001 - convert backend errors to contract failure
        raise ValueError(f"existing archive {path} cannot be decompressed") from error
    if (
        digest.hexdigest() != expected.sha256
        or byte_count != expected.byte_count
        or newline_count != len(expected.ids)
        or last_byte != b"\n"
    ):
        raise ValueError(f"existing archive {path} does not round-trip to the raw shard")
    return _ArchiveScan(
        archive_bytes=path.stat().st_size,
        archive_sha256=_sha256_file(path),
        decompressed_bytes=byte_count,
        decompressed_sha256=digest.hexdigest(),
        decompressed_records=newline_count,
    )


def _resume_without_source(
    *,
    archive_path: Path,
    seal_manifest_path: Path,
    archive_manifest_path: Path,
    expected_records: int | None,
) -> RawShardSealSummary:
    if not all(
        path.is_file()
        for path in (archive_path, seal_manifest_path, archive_manifest_path)
    ):
        raise FileNotFoundError("raw source is absent and verified archive artifacts are incomplete")
    seal = _read_object(seal_manifest_path)
    manifest = _read_object(archive_manifest_path)
    if seal.get("schema_version") != RAW_SHARD_SEAL_SCHEMA_VERSION:
        raise ValueError("seal manifest schema_version is incompatible")
    if manifest.get("schema_version") != RAW_SHARD_ARCHIVE_SCHEMA_VERSION:
        raise ValueError("archive manifest schema_version is incompatible")
    if seal.get("downstream_acknowledged") is not True:
        raise ValueError("seal manifest does not acknowledge downstream queues")
    source = seal.get("source")
    seal_reference = manifest.get("seal_manifest")
    archive = manifest.get("archive")
    verification = manifest.get("verification")
    if not all(
        isinstance(value, Mapping)
        for value in (source, seal_reference, archive, verification)
    ):
        raise ValueError("archive manifests are malformed")
    records = _required_int(source, "records")
    if expected_records is not None and records != expected_records:
        raise ValueError("sealed record count does not match expected_records")
    source_bytes = _required_int(source, "bytes")
    source_sha256 = source.get("sha256")
    if not isinstance(source_sha256, str) or len(source_sha256) != 64:
        raise ValueError("seal manifest source sha256 is invalid")
    if seal_reference.get("sha256") != _sha256_file(seal_manifest_path):
        raise ValueError("archive manifest seal reference does not match seal manifest")
    archive_bytes = _required_int(archive, "bytes")
    archive_sha256 = archive.get("sha256")
    if archive_bytes != archive_path.stat().st_size:
        raise ValueError(f"existing archive {archive_path} size does not match manifest")
    if _sha256_file(archive_path) != archive_sha256:
        raise ValueError(f"existing archive {archive_path} sha256 does not match manifest")
    if (
        verification.get("decompressed_sha256") != source_sha256
        or verification.get("decompressed_bytes") != source_bytes
        or verification.get("decompressed_records") != records
    ):
        raise ValueError("archive manifest verification does not match sealed source")
    return RawShardSealSummary(
        status="unchanged",
        records=records,
        successful_records=_required_int(source, "successful_records"),
        api_error_records=_required_int(source, "api_error_records"),
        source_bytes=source_bytes,
        source_sha256=source_sha256,
        archive_bytes=archive_bytes,
        archive_sha256=str(archive_sha256),
        source_removed=True,
    )


def _read_unique_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number}: expected an object")
            record_id = value.get("id")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"{path}:{line_number}: invalid id")
            if record_id in ids:
                raise ValueError(f"{path}:{line_number}: duplicate id {record_id!r}")
            ids.add(record_id)
    return ids


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected an object")
    return value


def _required_int(value: Mapping[str, Any], key: str) -> int:
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, int) or result < 0:
        raise ValueError(f"manifest field {key!r} must be a non-negative integer")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_zstandard() -> Any:
    try:
        import zstandard
    except ImportError as error:
        raise RuntimeError(
            'raw shard compression requires: pip install -e ".[archive]"'
        ) from error
    return zstandard
