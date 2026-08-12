"""Transformers trainer and collator for offline ALM."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from transformers import Trainer

from .alm import (
    aggregate_chunk_logprobs,
    alm_forward_kl_loss,
    causal_actual_token_logprobs,
)


@dataclass(slots=True)
class ALMDataCollator:
    """Right-pad SFT sequences and their independent ragged ALM chunk axis."""

    pad_token_id: int
    pad_to_multiple_of: int | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.pad_token_id, bool)
            or not isinstance(self.pad_token_id, int)
            or self.pad_token_id < 0
        ):
            raise ValueError("pad_token_id must be a non-negative integer")
        if self.pad_to_multiple_of is not None and (
            isinstance(self.pad_to_multiple_of, bool)
            or not isinstance(self.pad_to_multiple_of, int)
            or self.pad_to_multiple_of <= 0
        ):
            raise ValueError("pad_to_multiple_of must be a positive integer")

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("ALMDataCollator requires at least one feature")
        sequence_lengths = [len(_list_field(feature, "input_ids")) for feature in features]
        if any(length == 0 for length in sequence_lengths):
            raise ValueError("input_ids must not be empty")
        max_length = max(sequence_lengths)
        if self.pad_to_multiple_of is not None:
            max_length = (
                (max_length + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        padded_input_ids: list[list[int]] = []
        padded_attention_mask: list[list[int]] = []
        padded_labels: list[list[int]] = []
        padded_student_chunk_ids: list[list[int]] = []
        teacher_rows: list[list[float]] = []
        chunk_counts: list[int] = []

        for feature, length in zip(features, sequence_lengths, strict=True):
            input_ids = _integer_list(feature, "input_ids")
            attention_mask = _integer_list(feature, "attention_mask")
            labels = _integer_list(feature, "labels")
            student_chunk_ids = _integer_list(feature, "alm_student_chunk_ids")
            if not (
                len(attention_mask)
                == len(labels)
                == len(student_chunk_ids)
                == length
            ):
                raise ValueError("all sequence-valued ALM fields must match input_ids length")
            teacher_logprobs = _float_list(feature, "alm_teacher_chunk_logprobs")
            declared_count = feature.get("alm_chunk_count")
            if (
                isinstance(declared_count, bool)
                or not isinstance(declared_count, int)
                or declared_count < 0
            ):
                raise ValueError("alm_chunk_count must be a non-negative integer")
            if declared_count != len(teacher_logprobs):
                raise ValueError("alm_chunk_count must match teacher chunk logprobs")
            if any(chunk_id < -1 or chunk_id >= declared_count for chunk_id in student_chunk_ids):
                raise ValueError("alm_student_chunk_ids contains an out-of-range chunk")

            padding = max_length - length
            padded_input_ids.append(input_ids + [self.pad_token_id] * padding)
            padded_attention_mask.append(attention_mask + [0] * padding)
            padded_labels.append(labels + [-100] * padding)
            padded_student_chunk_ids.append(student_chunk_ids + [-1] * padding)
            teacher_rows.append(teacher_logprobs)
            chunk_counts.append(declared_count)

        max_chunks = max(chunk_counts)
        padded_teacher = [
            row + [0.0] * (max_chunks - len(row)) for row in teacher_rows
        ]
        chunk_mask = [
            [True] * count + [False] * (max_chunks - count) for count in chunk_counts
        ]
        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "alm_student_chunk_ids": torch.tensor(
                padded_student_chunk_ids, dtype=torch.long
            ),
            "alm_teacher_chunk_logprobs": torch.tensor(
                padded_teacher, dtype=torch.float32
            ),
            "alm_chunk_mask": torch.tensor(chunk_mask, dtype=torch.bool),
        }


def _list_field(feature: Mapping[str, Any], key: str) -> list[Any]:
    value = feature.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return value


def _integer_list(feature: Mapping[str, Any], key: str) -> list[int]:
    values = _list_field(feature, key)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError(f"{key} must contain integers")
    return list(values)


def _float_list(feature: Mapping[str, Any], key: str) -> list[float]:
    values = _list_field(feature, key)
    result: list[float] = []
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) > 1e-7
        ):
            raise ValueError(f"{key} must contain finite non-positive numbers")
        result.append(float(value))
    return result


class ALMTrainer(Trainer):
    """Causal SFT plus offline cross-tokenizer ALM for Transformers/PEFT models."""

    def __init__(
        self,
        *args: Any,
        alpha_alm: float = 1.0,
        alm_temperature: float = 100.0,
        alm_epsilon: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        if not math.isfinite(alpha_alm) or alpha_alm < 0:
            raise ValueError("alpha_alm must be finite and non-negative")
        if not math.isfinite(alm_temperature) or alm_temperature <= 0:
            raise ValueError("alm_temperature must be finite and positive")
        if not math.isfinite(alm_epsilon) or alm_epsilon < 0:
            raise ValueError("alm_epsilon must be finite and non-negative")
        super().__init__(*args, **kwargs)
        self.alpha_alm = float(alpha_alm)
        self.alm_temperature = float(alm_temperature)
        self.alm_epsilon = float(alm_epsilon)
        self._loss_component_sums: dict[str, torch.Tensor] = {}
        self._loss_component_count = 0

    def _record_loss_components(
        self,
        hard_sft_loss: torch.Tensor,
        alm_loss: torch.Tensor | None,
        combined_loss: torch.Tensor,
    ) -> None:
        components = {
            "hard_sft_loss": hard_sft_loss,
            "combined_loss": combined_loss,
        }
        if alm_loss is not None:
            components["alm_loss"] = alm_loss
            components["weighted_alm_loss"] = self.alpha_alm * alm_loss
        if not hasattr(self, "_loss_component_sums"):
            self._loss_component_sums = {}
            self._loss_component_count = 0
        for name, value in components.items():
            detached = value.detach().to(dtype=torch.float64)
            previous = self._loss_component_sums.get(name)
            self._loss_component_sums[name] = (
                detached if previous is None else previous + detached
            )
        self._loss_component_count += 1

    def _pop_loss_component_metrics(self) -> dict[str, float]:
        count = getattr(self, "_loss_component_count", 0)
        if count == 0:
            return {}
        metrics = {
            name: (total / count).item()
            for name, total in self._loss_component_sums.items()
        }
        self._loss_component_sums = {}
        self._loss_component_count = 0
        return metrics

    def log(
        self,
        logs: dict[str, float],
        start_time: float | None = None,
    ) -> None:
        if "loss" in logs:
            logs = {**logs, **self._pop_loss_component_metrics()}
        super().log(logs, start_time)

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        del num_items_in_batch
        required = (
            "input_ids",
            "alm_student_chunk_ids",
            "alm_teacher_chunk_logprobs",
            "alm_chunk_mask",
        )
        missing = [key for key in required if key not in inputs]
        if missing:
            raise ValueError(f"ALM batch is missing required fields: {', '.join(missing)}")

        model_inputs = {
            key: value for key, value in inputs.items() if not key.startswith("alm_")
        }
        outputs = model(**model_inputs)
        hard_sft_loss = (
            outputs["loss"]
            if isinstance(outputs, Mapping)
            else getattr(outputs, "loss", None)
        )
        logits = (
            outputs["logits"]
            if isinstance(outputs, Mapping)
            else getattr(outputs, "logits", None)
        )
        if hard_sft_loss is None or logits is None:
            raise ValueError("causal model output must contain both loss and logits")

        alm_loss: torch.Tensor | None = None
        if self.alpha_alm == 0.0:
            total_loss = hard_sft_loss
        else:
            student_token_logprobs = causal_actual_token_logprobs(
                logits, inputs["input_ids"]
            )
            teacher_chunk_logprobs = inputs["alm_teacher_chunk_logprobs"]
            student_chunk_logprobs = aggregate_chunk_logprobs(
                student_token_logprobs,
                inputs["alm_student_chunk_ids"][:, 1:],
                num_chunks=teacher_chunk_logprobs.shape[1],
            )
            alm_loss = alm_forward_kl_loss(
                teacher_chunk_logprobs,
                student_chunk_logprobs,
                inputs["alm_chunk_mask"],
                temperature=self.alm_temperature,
                epsilon=self.alm_epsilon,
            )
            total_loss = hard_sft_loss + self.alpha_alm * alm_loss
        if model.training:
            self._record_loss_components(hard_sft_loss, alm_loss, total_loss)

        if return_outputs:
            return total_loss, outputs
        return total_loss
