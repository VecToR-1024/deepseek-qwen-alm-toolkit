from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers import Trainer

from topk_distill.alm import (
    aggregate_chunk_logprobs,
    alm_forward_kl_loss,
    causal_actual_token_logprobs,
)
from topk_distill.alm_trainer import ALMDataCollator, ALMTrainer


def test_alm_data_collator_pads_sequences_and_chunk_axis_independently() -> None:
    collator = ALMDataCollator(pad_token_id=0)
    features = [
        {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
            "labels": [-100, 2, 3],
            "alm_student_chunk_ids": [-1, 0, 1],
            "alm_teacher_chunk_logprobs": [-0.2, -0.3],
            "alm_chunk_count": 2,
            "alm_dropped_boundary_chunks": 0,
        },
        {
            "input_ids": [4, 5],
            "attention_mask": [1, 1],
            "labels": [-100, 5],
            "alm_student_chunk_ids": [-1, 0],
            "alm_teacher_chunk_logprobs": [-0.4],
            "alm_chunk_count": 1,
            "alm_dropped_boundary_chunks": 0,
        },
    ]

    batch = collator(features)

    assert batch["input_ids"].tolist() == [[1, 2, 3], [4, 5, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert batch["labels"].tolist() == [[-100, 2, 3], [-100, 5, -100]]
    assert batch["alm_student_chunk_ids"].tolist() == [[-1, 0, 1], [-1, 0, -1]]
    assert batch["alm_teacher_chunk_logprobs"] == pytest.approx(
        torch.tensor([[-0.2, -0.3], [-0.4, 0.0]])
    )
    assert batch["alm_chunk_mask"].tolist() == [[True, True], [True, False]]


class TinyCausalModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(
            torch.tensor(
                [
                    [
                        [0.0, 2.0, -1.0],
                        [0.0, -2.0, 3.0],
                        [0.0, 0.0, 0.0],
                    ]
                ]
            )
        )

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> SimpleNamespace:
        assert input_ids.shape == attention_mask.shape == labels.shape
        return SimpleNamespace(loss=self.logits.sum() * 0.0 + 2.0, logits=self.logits)


def test_trainer_combines_hard_sft_and_alm_without_teacher_model() -> None:
    model = TinyCausalModel()
    trainer = object.__new__(ALMTrainer)
    trainer.alpha_alm = 0.5
    trainer.alm_temperature = 1.0
    trainer.alm_epsilon = 0.0
    inputs = {
        "input_ids": torch.tensor([[0, 1, 2]]),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
        "labels": torch.tensor([[-100, 1, 2]]),
        "alm_student_chunk_ids": torch.tensor([[-1, 0, 0]]),
        "alm_teacher_chunk_logprobs": torch.tensor([[-0.7]]),
        "alm_chunk_mask": torch.tensor([[True]]),
    }

    loss, outputs = trainer.compute_loss(model, inputs, return_outputs=True)

    student_tokens = causal_actual_token_logprobs(model.logits, inputs["input_ids"])
    student_chunks = aggregate_chunk_logprobs(
        student_tokens,
        inputs["alm_student_chunk_ids"][:, 1:],
        num_chunks=1,
    )
    expected_alm = alm_forward_kl_loss(
        inputs["alm_teacher_chunk_logprobs"],
        student_chunks,
        inputs["alm_chunk_mask"],
        temperature=1.0,
        epsilon=0.0,
    )
    assert loss.item() == pytest.approx(2.0 + 0.5 * expected_alm.item(), abs=1e-7)
    assert outputs.loss.item() == pytest.approx(2.0)
    loss.backward()
    assert model.logits.grad is not None
    assert torch.isfinite(model.logits.grad).all()


def test_trainer_logs_averaged_hard_sft_and_alm_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = TinyCausalModel()
    trainer = object.__new__(ALMTrainer)
    trainer.alpha_alm = 0.5
    trainer.alm_temperature = 1.0
    trainer.alm_epsilon = 0.0
    inputs = {
        "input_ids": torch.tensor([[0, 1, 2]]),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
        "labels": torch.tensor([[-100, 1, 2]]),
        "alm_student_chunk_ids": torch.tensor([[-1, 0, 0]]),
        "alm_teacher_chunk_logprobs": torch.tensor([[-0.7]]),
        "alm_chunk_mask": torch.tensor([[True]]),
    }
    captured: list[tuple[dict[str, float], float | None]] = []

    def capture_parent_log(
        self: Trainer,
        logs: dict[str, float],
        start_time: float | None = None,
    ) -> None:
        del self
        captured.append((dict(logs), start_time))

    monkeypatch.setattr(Trainer, "log", capture_parent_log)

    trainer.compute_loss(model, inputs)
    trainer.compute_loss(model, inputs)
    trainer.log({"loss": 3.0}, start_time=12.5)

    student_tokens = causal_actual_token_logprobs(model.logits, inputs["input_ids"])
    student_chunks = aggregate_chunk_logprobs(
        student_tokens,
        inputs["alm_student_chunk_ids"][:, 1:],
        num_chunks=1,
    )
    expected_alm = alm_forward_kl_loss(
        inputs["alm_teacher_chunk_logprobs"],
        student_chunks,
        inputs["alm_chunk_mask"],
        temperature=1.0,
        epsilon=0.0,
    ).item()
    assert captured == [
        (
            {
                "loss": 3.0,
                "hard_sft_loss": pytest.approx(2.0),
                "alm_loss": pytest.approx(expected_alm),
                "weighted_alm_loss": pytest.approx(0.5 * expected_alm),
                "combined_loss": pytest.approx(2.0 + 0.5 * expected_alm),
            },
            12.5,
        )
    ]

    trainer.log({"train_runtime": 1.0})
    assert captured[-1] == ({"train_runtime": 1.0}, None)
