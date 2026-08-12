#!/usr/bin/env python3
"""Package or upload an audited offline-ALM dataset release."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from deepseek_distill.hf_dataset_release import (
    audit_hf_package,
    package_hf_dataset,
    upload_hf_package,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    package = subparsers.add_parser(
        "package", help="Create deterministic, redacted JSONL gzip shards."
    )
    package.add_argument("--input", type=Path, required=True)
    package.add_argument("--output-dir", type=Path, required=True)
    package.add_argument("--config-name", required=True)
    package.add_argument("--repo-id", required=True, metavar="NAMESPACE/NAME")
    package.add_argument("--records-per-shard", type=int, default=250)
    package.add_argument("--expected-records", type=int)
    package.add_argument("--expected-sha256")

    audit = subparsers.add_parser(
        "audit", help="Revalidate shard hashes, trace reconstruction, and redactions."
    )
    audit.add_argument("--package-dir", type=Path, required=True)

    upload = subparsers.add_parser(
        "upload", help="Upload a previously packaged release (dry-run by default)."
    )
    upload.add_argument("--package-dir", type=Path, required=True)
    upload.add_argument("--repo-id", required=True, metavar="NAMESPACE/NAME")
    upload.add_argument(
        "--public",
        action="store_true",
        help="Create a public repo. The safer default is private.",
    )
    upload.add_argument(
        "--commit-message",
        default="Upload audited offline ALM training traces",
    )
    upload.add_argument(
        "--execute",
        action="store_true",
        help="Perform the network mutation; otherwise print the upload plan only.",
    )
    upload.add_argument(
        "--confirm-manifest-sha256",
        help="Human-reviewed release_manifest.json SHA256; required with --execute.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "package":
        manifest = package_hf_dataset(
            input_path=args.input,
            output_dir=args.output_dir,
            config_name=args.config_name,
            repo_id=args.repo_id,
            records_per_shard=args.records_per_shard,
            expected_records=args.expected_records,
            expected_sha256=args.expected_sha256,
        )
        result = {
            "event": "hf_dataset_package_created",
            "output_dir": str(args.output_dir),
            "records": manifest["counts"]["records"],
            "repo_id": args.repo_id,
            "shards": manifest["counts"]["shards"],
        }
    elif args.command == "audit":
        audit = audit_hf_package(args.package_dir)
        result = {"event": "hf_dataset_release_audited", **audit}
    else:
        manifest_path = args.package_dir / "release_manifest.json"
        if not manifest_path.is_file():
            raise ValueError("package_dir does not contain release_manifest.json")
        manifest_sha256 = _sha256(manifest_path)
        if not args.execute:
            result = {
                "event": "hf_dataset_upload_plan",
                "execute": False,
                "manifest_sha256": manifest_sha256,
                "package_dir": str(args.package_dir),
                "repo_id": args.repo_id,
                "visibility": "public" if args.public else "private",
            }
        else:
            if not args.confirm_manifest_sha256:
                raise ValueError(
                    "--execute requires --confirm-manifest-sha256 from human review"
                )
            upload = upload_hf_package(
                package_dir=args.package_dir,
                repo_id=args.repo_id,
                private=not args.public,
                confirmed_manifest_sha256=args.confirm_manifest_sha256,
                commit_message=args.commit_message,
            )
            result = {"event": "hf_dataset_upload_completed", **upload}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
