#!/usr/bin/env python3
"""Wait for a frozen hard supplement, upload it atomically, and start training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
import shlex
import sys
import time
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-frozen-dir", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--remote-upload-root", required=True)
    parser.add_argument("--remote-run-root", required=True)
    parser.add_argument("--wait-timeout", type=float, default=14_400.0)
    parser.add_argument("--poll-interval", type=float, default=15.0)
    parser.add_argument("--status-output", type=Path)
    return parser


def validate_frozen_dataset(frozen_dir: Path) -> dict[str, Any]:
    frozen_dir = Path(frozen_dir)
    data_path = frozen_dir / "training_records.jsonl"
    manifest_path = frozen_dir / "dataset_manifest.json"
    if not data_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"frozen dataset is not ready: {frozen_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "offline_alm.frozen_multisource_training.v1":
        raise ValueError("unsupported frozen dataset manifest schema")
    if manifest.get("training_started") is not False:
        raise ValueError("frozen dataset manifest has an invalid training_started flag")
    output = manifest.get("outputs", {}).get("training_records", {})
    expected_records = output.get("records")
    expected_bytes = output.get("bytes")
    expected_sha = output.get("sha256")
    if not isinstance(expected_records, int) or expected_records <= 0:
        raise ValueError("manifest has no positive training record count")

    digest = hashlib.sha256()
    records = 0
    bytes_read = 0
    with data_path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            bytes_read += len(line)
            if line.strip():
                records += 1
    actual_sha = digest.hexdigest()
    if records != expected_records:
        raise ValueError(
            f"training record count mismatch: actual={records}, expected={expected_records}"
        )
    if bytes_read != expected_bytes:
        raise ValueError(
            f"training byte-size mismatch: actual={bytes_read}, expected={expected_bytes}"
        )
    if actual_sha != expected_sha:
        raise ValueError("training SHA-256 does not match the frozen manifest")
    return {
        "data_path": data_path,
        "manifest_path": manifest_path,
        "records": records,
        "data_bytes": bytes_read,
        "data_sha256": actual_sha,
        "manifest_sha256": _sha256_file(manifest_path),
    }


def build_remote_start_command(*, upload_root: str, run_root: str) -> str:
    upload = shlex.quote(upload_root)
    run = shlex.quote(run_root)
    return " ".join(
        [
            "set -euo pipefail;",
            f"upload={upload};",
            f"run={run};",
            "if [[ ! -f \"$upload/training_records.jsonl\" || ! -f \"$upload/dataset_manifest.json\" || ! -f \"$upload/.deployment_complete\" ]]; then",
            "echo refusing_existing_upload_state >&2; exit 2; fi;",
            "if [[ -e \"$run/training_manifest.json\" || -e \"$run/outputs\" || -e \"$run/smoke\" || -e \"$run/supervisor.pid\" ]]; then",
            "echo refusing_started_run >&2; exit 2; fi;",
            "bash -n \"$run/prepare_and_launch.sh\";",
            "bash -n \"$run/monitor_training.sh\";",
            "nohup bash \"$run/prepare_and_launch.sh\" > \"$run/supervisor.log\" 2>&1 < /dev/null &",
            "pid=$!; printf '%s\\n' \"$pid\" > \"$run/supervisor.pid\";",
            "echo training_supervisor_pid=$pid;",
        ]
    )


def wait_for_frozen_dataset(
    frozen_dir: Path,
    *,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            return validate_frozen_dataset(frozen_dir)
        except FileNotFoundError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"frozen dataset did not appear: {frozen_dir}")
            print(
                json.dumps(
                    {"event": "waiting_for_frozen_dataset", "path": str(frozen_dir)},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            time.sleep(poll_interval_seconds)


def deploy(args: argparse.Namespace) -> dict[str, Any]:
    password = os.environ.get("REMOTE_SSH_PASSWORD")
    if not password:
        raise RuntimeError("REMOTE_SSH_PASSWORD is required")
    validated = wait_for_frozen_dataset(
        args.local_frozen_dir,
        timeout_seconds=args.wait_timeout,
        poll_interval_seconds=args.poll_interval,
    )
    print(
        json.dumps(
            {
                "event": "local_frozen_dataset_validated",
                "records": validated["records"],
                "bytes": validated["data_bytes"],
                "sha256": validated["data_sha256"],
            }
        ),
        flush=True,
    )

    import paramiko

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        args.host,
        port=args.port,
        username=args.user,
        password=password,
        timeout=20,
        allow_agent=False,
        look_for_keys=False,
    )
    try:
        _prepare_remote_upload_root(client, args.remote_upload_root)
        with client.open_sftp() as sftp:
            _upload_atomic(
                sftp,
                validated["data_path"],
                posixpath.join(args.remote_upload_root, "training_records.jsonl"),
            )
            _upload_atomic(
                sftp,
                validated["manifest_path"],
                posixpath.join(args.remote_upload_root, "dataset_manifest.json"),
            )
        _attest_remote_upload(client, args.remote_upload_root, validated)
        start_output = _exec_checked(
            client,
            build_remote_start_command(
                upload_root=args.remote_upload_root,
                run_root=args.remote_run_root,
            ),
        ).strip()
    finally:
        client.close()

    result = {
        "event": "hard_training_deployed",
        "records": validated["records"],
        "data_sha256": validated["data_sha256"],
        "remote_upload_root": args.remote_upload_root,
        "remote_run_root": args.remote_run_root,
        "remote_start": start_output,
        "training_contract": "preflight_then_one_step_smoke_then_alpha10_two_epochs",
    }
    if args.status_output is not None:
        args.status_output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.status_output.with_suffix(args.status_output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, args.status_output)
    return result


def _prepare_remote_upload_root(client: Any, upload_root: str) -> None:
    quoted = shlex.quote(upload_root)
    command = (
        "set -e; "
        f"if test -e {quoted}; then echo remote upload root already exists >&2; exit 2; fi; "
        f"mkdir -p {quoted}"
    )
    _exec_checked(client, command)


def _upload_atomic(sftp: Any, local_path: Path, remote_path: str) -> None:
    partial = remote_path + ".partial"
    size = local_path.stat().st_size
    last_reported = -1

    def progress(transferred: int, total: int) -> None:
        nonlocal last_reported
        percent = int(100 * transferred / total) if total else 100
        bucket = percent // 10
        if bucket != last_reported:
            last_reported = bucket
            print(
                json.dumps(
                    {
                        "event": "upload_progress",
                        "file": local_path.name,
                        "percent": min(percent, 100),
                    }
                ),
                flush=True,
            )

    sftp.put(str(local_path), partial, callback=progress, confirm=True)
    if sftp.stat(partial).st_size != size:
        raise IOError(f"remote partial upload has the wrong size: {partial}")
    sftp.rename(partial, remote_path)


def _attest_remote_upload(
    client: Any,
    upload_root: str,
    validated: dict[str, Any],
) -> None:
    data = shlex.quote(posixpath.join(upload_root, "training_records.jsonl"))
    manifest = shlex.quote(posixpath.join(upload_root, "dataset_manifest.json"))
    marker = shlex.quote(posixpath.join(upload_root, ".deployment_complete"))
    command = (
        "set -e; "
        f"test \"$(stat -c %s {data})\" = {validated['data_bytes']}; "
        f"test \"$(sha256sum {data} | cut -d' ' -f1)\" = {validated['data_sha256']}; "
        f"test \"$(sha256sum {manifest} | cut -d' ' -f1)\" = {validated['manifest_sha256']}; "
        f"printf '%s\\n' {validated['data_sha256']} > {marker}"
    )
    _exec_checked(client, command)


def _exec_checked(client: Any, command: str) -> str:
    _, stdout, stderr = client.exec_command(command)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    status = stdout.channel.recv_exit_status()
    if status != 0:
        raise RuntimeError(f"remote command failed with exit {status}: {err.strip()}")
    if err:
        print(err, file=sys.stderr, end="", flush=True)
    return out


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = build_parser().parse_args()
    if args.wait_timeout <= 0 or args.poll_interval <= 0:
        raise ValueError("wait timeout and poll interval must be positive")
    print(json.dumps(deploy(args), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
