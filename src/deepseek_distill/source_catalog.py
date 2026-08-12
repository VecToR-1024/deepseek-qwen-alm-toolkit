"""Frozen registry for supported external coding-data sources."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .apps import (
    APPS_CONFIG,
    APPS_DATASET_ID,
    APPS_LICENSE,
    APPS_MIRROR,
    APPS_PROVENANCE,
    APPS_REVISION,
    APPS_SPLIT,
    load_apps_tasks,
)
from .code_contests import (
    CODE_CONTESTS_CONFIG,
    CODE_CONTESTS_DATASET_ID,
    CODE_CONTESTS_LICENSE,
    CODE_CONTESTS_MIRROR,
    CODE_CONTESTS_PROVENANCE,
    CODE_CONTESTS_REVISION,
    CODE_CONTESTS_SPLIT,
    load_code_contests_tasks,
)
from .odex import (
    ODEX_CONFIG,
    ODEX_DATASET_ID,
    ODEX_LICENSE,
    ODEX_MIRROR,
    ODEX_PROVENANCE,
    ODEX_REVISION,
    ODEX_SPLIT,
    load_odex_tasks,
)
from .open_r1_codeforces import (
    OPEN_R1_CODEFORCES_CONFIG,
    OPEN_R1_CODEFORCES_DATASET_ID,
    OPEN_R1_CODEFORCES_LICENSE,
    OPEN_R1_CODEFORCES_MIRROR,
    OPEN_R1_CODEFORCES_PROVENANCE,
    OPEN_R1_CODEFORCES_REVISION,
    OPEN_R1_CODEFORCES_SPLIT,
    load_open_r1_codeforces_tasks,
)
from .taco import (
    TACO_CARD,
    TACO_DATASET_ID,
    TACO_LICENSE,
    TACO_PROVENANCE,
    TACO_REVISION,
    TACO_SPLIT,
)
from .taco_multishard import (
    TACO_MULTISHARD_CONFIG,
    load_taco_multishard_tasks,
)
from .xcodeeval import (
    XCODEEVAL_CONFIG,
    XCODEEVAL_DATASET_ID,
    XCODEEVAL_LICENSE,
    XCODEEVAL_MIRROR,
    XCODEEVAL_PROVENANCE,
    XCODEEVAL_REVISION,
    XCODEEVAL_SPLIT,
    load_xcodeeval_tasks,
)


Loader = Callable[..., list[dict[str, Any]]]


@dataclass(frozen=True)
class SourceSpec:
    key: str
    dataset_id: str
    config: str
    split: str
    revision: str
    license: str
    provenance: str
    mirror: str
    notes: tuple[str, ...]
    loader: Loader


SOURCE_SPECS = {
    spec.key: spec
    for spec in (
        SourceSpec(
            key="apps",
            dataset_id=APPS_DATASET_ID,
            config=APPS_CONFIG,
            split=APPS_SPLIT,
            revision=APPS_REVISION,
            license=APPS_LICENSE,
            provenance=APPS_PROVENANCE,
            mirror=APPS_MIRROR,
            notes=(
                "official_train_split_only",
                "pure_stdin_stdout_rows_without_starter_code",
            ),
            loader=load_apps_tasks,
        ),
        SourceSpec(
            key="code-contests",
            dataset_id=CODE_CONTESTS_DATASET_ID,
            config=CODE_CONTESTS_CONFIG,
            split=CODE_CONTESTS_SPLIT,
            revision=CODE_CONTESTS_REVISION,
            license=CODE_CONTESTS_LICENSE,
            provenance=CODE_CONTESTS_PROVENANCE,
            mirror=CODE_CONTESTS_MIRROR,
            notes=(
                "official_train_split_only",
                "bounded_memory_streaming_selection",
                "public_private_generated_tests_retained_outside_prompt",
            ),
            loader=load_code_contests_tasks,
        ),
        SourceSpec(
            key="odex",
            dataset_id=ODEX_DATASET_ID,
            config=ODEX_CONFIG,
            split=ODEX_SPLIT,
            revision=ODEX_REVISION,
            license=ODEX_LICENSE,
            provenance=ODEX_PROVENANCE,
            mirror=ODEX_MIRROR,
            notes=(
                "official_dataset_exposes_only_test_split",
                "test_split_used_as_training_source_by_explicit_project_decision",
                "only_standard_library_safe_function_tasks",
                "eligible_unique_pool_observed_locally_as_207",
            ),
            loader=load_odex_tasks,
        ),
        SourceSpec(
            key="open-r1-codeforces",
            dataset_id=OPEN_R1_CODEFORCES_DATASET_ID,
            config=OPEN_R1_CODEFORCES_CONFIG,
            split=OPEN_R1_CODEFORCES_SPLIT,
            revision=OPEN_R1_CODEFORCES_REVISION,
            license=OPEN_R1_CODEFORCES_LICENSE,
            provenance=OPEN_R1_CODEFORCES_PROVENANCE,
            mirror=OPEN_R1_CODEFORCES_MIRROR,
            notes=(
                "official_train_split_only",
                "verifiable_subset_only",
                "executable_complete_official_tests_only",
                "stdio_exact_output_without_custom_checker_only",
                "interactive_and_file_io_tasks_excluded",
                "generated_tests_not_downloaded",
                "dataset_tag_says_cc_by_4_0_readme_body_says_odc_by_4_0",
            ),
            loader=load_open_r1_codeforces_tasks,
        ),
        SourceSpec(
            key="xcodeeval",
            dataset_id=XCODEEVAL_DATASET_ID,
            config=XCODEEVAL_CONFIG,
            split=XCODEEVAL_SPLIT,
            revision=XCODEEVAL_REVISION,
            license=XCODEEVAL_LICENSE,
            provenance=XCODEEVAL_PROVENANCE,
            mirror=XCODEEVAL_MIRROR,
            notes=(
                "compact_split_used_as_training_source_by_explicit_project_decision",
                "compact_rows_come_from_raw_validation_file",
                "official_hidden_tests_retained_outside_prompt",
                "eligible_unique_compact_pool_observed_locally_as_106",
                "pinned_loader_and_dataset_card_license_labels_conflict",
            ),
            loader=load_xcodeeval_tasks,
        ),
        SourceSpec(
            key="taco-multishard",
            dataset_id=TACO_DATASET_ID,
            config=TACO_MULTISHARD_CONFIG,
            split=TACO_SPLIT,
            revision=TACO_REVISION,
            license=TACO_LICENSE,
            provenance=TACO_PROVENANCE,
            mirror=TACO_CARD,
            notes=(
                "official_train_shards_1_through_8_only",
                "previously_attempted_shard_zero_excluded_by_construction",
                "pure_stdin_stdout_rows_without_starter_code_or_pictures",
                "geeksforgeeks_and_hackerrank_sources_excluded",
            ),
            loader=load_taco_multishard_tasks,
        ),
    )
}
