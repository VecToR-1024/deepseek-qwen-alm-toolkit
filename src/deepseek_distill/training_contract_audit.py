"""Aggregate the production ALM preprocessing contract over a dataset."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from .training_data_audit import (
    audit_labeled_sequence,
    distribution,
    ending_features,
    record_source_name,
    response_style_features,
)


def tokenizer_contract(tokenizer: Any) -> dict[str, Any]:
    """Capture the token IDs and exact chat-template identity."""

    template = getattr(tokenizer, "chat_template", None)
    return {
        "tokenizer_class": type(tokenizer).__name__,
        "eos_token": getattr(tokenizer, "eos_token", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token": getattr(tokenizer, "pad_token", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "bos_token": getattr(tokenizer, "bos_token", None),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "chat_template_sha256": (
            hashlib.sha256(template.encode("utf-8")).hexdigest()
            if isinstance(template, str)
            else None
        ),
        "chat_template": template,
    }


def build_training_contract_report(
    records: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    *,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the production ALM example builder and aggregate label/style facts."""

    from .alm_preprocessing import ALMExampleBuilder

    template_kwargs = dict(chat_template_kwargs or {})
    builder = ALMExampleBuilder(
        tokenizer,
        chat_template_kwargs=template_kwargs,
    )
    eos_value = getattr(tokenizer, "eos_token_id", None)
    eos_token_ids = (
        [int(item) for item in eos_value]
        if isinstance(eos_value, (list, tuple))
        else ([int(eos_value)] if eos_value is not None else [])
    )
    source_counts: Counter[str] = Counter()
    source_details: dict[str, dict[str, Any]] = {}
    suffixes: Counter[str] = Counter()
    last_teacher_tokens: Counter[str] = Counter()
    last_supervised_tokens: Counter[str] = Counter()
    last_input_tokens: Counter[str] = Counter()
    ending_characters: Counter[str] = Counter()
    lengths: dict[str, list[float | int]] = {
        "character_count": [],
        "utf8_byte_count": [],
        "line_count": [],
        "provider_token_count": [],
        "qwen_sequence_length": [],
        "qwen_supervised_token_count": [],
        "alm_chunk_count": [],
        "alm_boundary_drop_count": [],
        "comment_line_ratio": [],
        "docstring_line_ratio": [],
    }
    record_count = 0
    eos_present = 0
    eos_supervised = 0
    all_eos_supervised = 0
    comments = 0
    docstrings = 0
    fences = 0
    newline_endings = 0
    missing_eos_ids: list[str] = []
    ignored_eos_ids: list[str] = []
    invalid_template_boundary_ids: list[str] = []
    zero_chunk_ids: list[str] = []

    for record in records:
        record_count += 1
        record_id = str(record.get("id", f"record_{record_count}"))
        source_name = record_source_name(record)
        source_counts[source_name] += 1
        per_source = source_details.setdefault(
            source_name,
            {
                "records": 0,
                "records_with_comments": 0,
                "records_with_docstrings": 0,
                "records_with_code_fences": 0,
                "records_ending_with_newline": 0,
                "provider_token_counts": [],
                "qwen_supervised_token_counts": [],
            },
        )
        per_source["records"] += 1
        response_text = record.get("response_text")
        content_tokens = record.get("content_tokens")
        if not isinstance(response_text, str) or not isinstance(content_tokens, list):
            raise ValueError(f"{record_id}: invalid response_text/content_tokens")

        style = response_style_features(response_text)
        ending = ending_features(response_text, content_tokens)
        example = builder.build(record)
        sequence = audit_labeled_sequence(
            input_ids=example["input_ids"],
            labels=example["labels"],
            eos_token_ids=eos_token_ids,
            id_to_token=lambda token_id: str(
                tokenizer.convert_ids_to_tokens(token_id)
            ),
        )

        messages = record["request"]["messages"]
        context = tokenizer.apply_chat_template(
            [dict(message) for message in messages],
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        full_messages = [dict(message) for message in messages]
        full_messages.append({"role": "assistant", "content": response_text})
        full_text = tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
            **template_kwargs,
        )
        expected_prefix = context + response_text
        if full_text.startswith(expected_prefix):
            suffixes[full_text[len(expected_prefix) :]] += 1
        else:
            invalid_template_boundary_ids.append(record_id)

        eos_present += int(sequence["eos_present"])
        eos_supervised += int(sequence["eos_supervised"])
        all_eos_supervised += int(sequence["all_eos_supervised"])
        if not sequence["eos_present"]:
            missing_eos_ids.append(record_id)
        elif not sequence["eos_supervised"]:
            ignored_eos_ids.append(record_id)
        comments += int(style["comment_token_count"] > 0)
        docstrings += int(style["docstring_line_count"] > 0)
        fences += int(style["has_code_fence"])
        newline_endings += int(ending["ends_with_newline"])
        per_source["records_with_comments"] += int(
            style["comment_token_count"] > 0
        )
        per_source["records_with_docstrings"] += int(
            style["docstring_line_count"] > 0
        )
        per_source["records_with_code_fences"] += int(style["has_code_fence"])
        per_source["records_ending_with_newline"] += int(
            ending["ends_with_newline"]
        )
        per_source["provider_token_counts"].append(len(content_tokens))
        per_source["qwen_supervised_token_counts"].append(
            sequence["supervised_token_count"]
        )

        for key in (
            "character_count",
            "utf8_byte_count",
            "line_count",
            "comment_line_ratio",
            "docstring_line_ratio",
        ):
            lengths[key].append(style[key])
        lengths["provider_token_count"].append(len(content_tokens))
        lengths["qwen_sequence_length"].append(sequence["sequence_length"])
        lengths["qwen_supervised_token_count"].append(
            sequence["supervised_token_count"]
        )
        chunk_count = int(example["alm_chunk_count"])
        boundary_drop_count = int(example["alm_dropped_boundary_chunks"])
        lengths["alm_chunk_count"].append(chunk_count)
        lengths["alm_boundary_drop_count"].append(boundary_drop_count)
        if chunk_count == 0:
            zero_chunk_ids.append(record_id)
        last_teacher = ending["last_teacher_token"]
        if last_teacher is not None:
            last_teacher_tokens[
                f"{last_teacher['bytes_hex']}|{last_teacher['utf8']!r}"
            ] += 1
        last_supervised = sequence["last_supervised_token"]
        if last_supervised is not None:
            last_supervised_tokens[
                f"{last_supervised['id']}|{last_supervised['token']!r}"
            ] += 1
        last_input = sequence["last_input_token"]
        if last_input is not None:
            last_input_tokens[
                f"{last_input['id']}|{last_input['token']!r}"
            ] += 1
        ending_characters[ending["last_non_whitespace_character"] or "<EMPTY>"] += 1

    return {
        "schema_version": "offline_alm.training_contract_audit.v1",
        "records": record_count,
        "chat_template_kwargs": template_kwargs,
        "sources": dict(sorted(source_counts.items())),
        "tokenizer": tokenizer_contract(tokenizer),
        "end_token_supervision": {
            "eos_present_records": eos_present,
            "eos_supervised_records": eos_supervised,
            "all_eos_supervised_records": all_eos_supervised,
            "missing_eos_record_ids": missing_eos_ids,
            "ignored_eos_record_ids": ignored_eos_ids,
            "assistant_suffixes": dict(suffixes.most_common()),
            "template_boundary_failure_record_ids": invalid_template_boundary_ids,
        },
        "alm_preprocessing": {
            "total_chunks": int(sum(lengths["alm_chunk_count"])),
            "chunk_count_distribution": distribution(lengths["alm_chunk_count"]),
            "boundary_drops": int(sum(lengths["alm_boundary_drop_count"])),
            "boundary_drop_distribution": distribution(
                lengths["alm_boundary_drop_count"]
            ),
            "zero_chunk_records": len(zero_chunk_ids),
            "zero_chunk_record_ids": zero_chunk_ids,
        },
        "teacher_response": {
            "records_with_comments": comments,
            "records_with_docstrings": docstrings,
            "records_with_code_fences": fences,
            "records_ending_with_newline": newline_endings,
            "distributions": {
                key: distribution(values) for key, values in lengths.items()
            },
            "last_teacher_token_distribution": dict(
                last_teacher_tokens.most_common()
            ),
            "last_supervised_student_token_distribution": dict(
                last_supervised_tokens.most_common()
            ),
            "last_input_token_distribution": dict(last_input_tokens.most_common()),
            "last_non_whitespace_character_distribution": dict(
                ending_characters.most_common()
            ),
            "by_source": {
                source_name: {
                    key: value
                    for key, value in details.items()
                    if key
                    not in {
                        "provider_token_counts",
                        "qwen_supervised_token_counts",
                    }
                }
                | {
                    "provider_token_count_distribution": distribution(
                        details["provider_token_counts"]
                    ),
                    "qwen_supervised_token_count_distribution": distribution(
                        details["qwen_supervised_token_counts"]
                    ),
                }
                for source_name, details in sorted(source_details.items())
            },
        },
    }
