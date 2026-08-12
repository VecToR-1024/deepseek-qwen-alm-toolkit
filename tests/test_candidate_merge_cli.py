from __future__ import annotations

from scripts.build_merged_candidates import build_parser, render_report_markdown


def test_merge_cli_freezes_authoritative_source_counts_and_max_length() -> None:
    args = build_parser().parse_args(
        [
            "--mbpp",
            "mbpp.jsonl",
            "--taco-pilot-retry",
            "taco49.jsonl",
            "--taco-breadth",
            "taco412.jsonl",
            "--output-dir",
            "merged",
        ]
    )

    assert args.expected_mbpp == 200
    assert args.expected_taco_pilot_retry == 49
    assert args.expected_taco_breadth == 412
    assert args.max_length == 4096
    assert args.student_tokenizer == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert args.student_revision == "c03e6d358207e414f1eca0bb1891e29f1db0e242"


def test_merge_report_describes_all_trainable_and_excluded_counts() -> None:
    report = render_report_markdown(
        {
            "counts": {
                "all_candidates": 661,
                "trainable_max4096": 655,
                "excluded": 6,
                "mbpp": 200,
                "taco": 461,
            },
            "source_order": ["mbpp200", "taco49", "taco412"],
            "exclusion_counts": {"sequence_length_exceeds_4096": 6},
            "duplicates": {"record_ids": 0, "problem_ids": 0},
            "alm": {
                "preprocessing_errors": 0,
                "zero_valid_chunks": 0,
                "records_exceeding_max_length": 6,
                "sequence_length_distribution": {"median": 900},
                "chunks_per_example_distribution": {"median": 200},
                "group_counts": {"1:1": 10, "1:N": 1, "N:1": 2, "N:M": 3},
                "prompt_completion_boundary_drops": 0,
            },
            "outputs": {
                "all_candidates": {"sha256": "a"},
                "trainable_max4096": {"sha256": "b"},
                "excluded_records": {"sha256": "c"},
            },
        }
    )

    assert "661" in report
    assert "655" in report
    assert "MBPP: 200" in report
    assert "TACO: 461" in report
    assert "sequence_length_exceeds_4096" in report
