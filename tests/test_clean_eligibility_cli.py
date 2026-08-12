from __future__ import annotations

from scripts.audit_clean_training_data import build_parser


def test_clean_audit_cli_requires_explicit_versioned_inputs_and_output() -> None:
    args = build_parser().parse_args(
        [
            "--training-data",
            "trainable.jsonl",
            "--alm-diagnostics",
            "alm.json",
            "--eos-attestation",
            "eos.json",
            "--output-dir",
            "clean-v4",
        ]
    )

    assert str(args.training_data) == "trainable.jsonl"
    assert str(args.alm_diagnostics) == "alm.json"
    assert str(args.eos_attestation) == "eos.json"
    assert str(args.output_dir) == "clean-v4"
    assert args.max_comment_line_ratio == 0.2
    assert args.max_sequence_length == 4096
