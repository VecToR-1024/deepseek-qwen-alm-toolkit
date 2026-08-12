from __future__ import annotations

import argparse
from pathlib import Path

import scripts.align_tokenizers as align_script
from deepseek_distill.alignment_pipeline import AlignmentDatasetSummary


def test_alignment_diagnostics_flag_is_opt_in() -> None:
    default_args = align_script.parse_args(["--input", "in.jsonl", "--output", "out.jsonl"])
    enabled_args = align_script.parse_args(
        ["--input", "in.jsonl", "--output", "out.jsonl", "--alignment-diagnostics"]
    )

    assert default_args.alignment_diagnostics is False
    assert enabled_args.alignment_diagnostics is True


def test_main_builds_span_aligner_only_when_diagnostics_are_enabled(
    monkeypatch,
) -> None:
    args = argparse.Namespace(
        input=Path("in.jsonl"),
        output=Path("out.jsonl"),
        student_model="fake-qwen",
        tokenizer=None,
        revision=None,
        cache_dir=None,
        local_files_only=True,
        force=False,
        alignment_diagnostics=True,
    )
    tokenizer = object()
    captured: dict = {}
    monkeypatch.setattr(align_script, "parse_args", lambda: args)
    monkeypatch.setattr(align_script, "load_tokenizer", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(
        align_script,
        "HuggingFaceByteOffsetTokenizer",
        lambda value: ("offset-adapter", value),
    )
    monkeypatch.setattr(
        align_script,
        "CrossTokenizerAligner",
        lambda value: ("span-aligner", value),
    )

    def fake_align_jsonl(*args, **kwargs):
        captured.update(kwargs)
        return AlignmentDatasetSummary(
            total_records=1,
            records_with_alignment=1,
            records_without_alignment=0,
            teacher_positions=1,
            aligned_positions=1,
            aligned_position_ratio=1.0,
        )

    monkeypatch.setattr(align_script, "align_jsonl", fake_align_jsonl)

    assert align_script.main() == 0
    assert captured["span_aligner"] == (
        "span-aligner",
        ("offset-adapter", tokenizer),
    )
