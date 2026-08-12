from __future__ import annotations

import math

import pytest
import torch

from topk_distill.alm import (
    aggregate_chunk_logprobs,
    alm_forward_kl_loss,
    binary_forward_kl_from_logprobs,
    causal_actual_token_logprobs,
)


def test_causal_actual_token_logprobs_use_previous_position_logits() -> None:
    input_ids = torch.tensor([[0, 1, 2]])
    logits = torch.tensor(
        [
            [
                [0.0, 2.0, -1.0],
                [0.0, -2.0, 3.0],
                [100.0, -100.0, -100.0],
            ]
        ]
    )

    actual = causal_actual_token_logprobs(logits, input_ids)

    expected_first = 2.0 - torch.logsumexp(torch.tensor([0.0, 2.0, -1.0]), dim=0)
    expected_second = 3.0 - torch.logsumexp(torch.tensor([0.0, -2.0, 3.0]), dim=0)
    assert actual.shape == (1, 2)
    assert actual[0] == pytest.approx(
        torch.tensor([expected_first, expected_second]), abs=1e-7
    )


def test_chunk_log_likelihood_is_sum_of_member_token_logprobs() -> None:
    token_logprobs = torch.tensor([[-0.1, -0.2, -0.3, -0.4]])
    shifted_chunk_ids = torch.tensor([[0, 0, 1, -1]])

    chunks = aggregate_chunk_logprobs(
        token_logprobs,
        shifted_chunk_ids,
        num_chunks=2,
    )

    assert chunks == pytest.approx(torch.tensor([[-0.3, -0.3]]), abs=1e-7)


def test_binary_forward_kl_matches_hand_computed_bernoulli_kl() -> None:
    teacher = torch.tensor([math.log(0.8)])
    student = torch.tensor([math.log(0.5)])

    result = binary_forward_kl_from_logprobs(
        teacher,
        student,
        temperature=1.0,
        epsilon=0.0,
    )

    expected = 0.8 * math.log(0.8 / 0.5) + 0.2 * math.log(0.2 / 0.5)
    assert result.item() == pytest.approx(expected, abs=1e-7)


def test_alm_loss_averages_only_valid_chunks() -> None:
    teacher = torch.log(torch.tensor([[0.8, 0.25], [0.6, 0.9]]))
    student = torch.log(torch.tensor([[0.5, 0.5], [0.4, 0.1]]))
    mask = torch.tensor([[True, False], [True, False]])

    loss = alm_forward_kl_loss(
        teacher,
        student,
        mask,
        temperature=1.0,
        epsilon=0.0,
    )

    first = 0.8 * math.log(0.8 / 0.5) + 0.2 * math.log(0.2 / 0.5)
    second = 0.6 * math.log(0.6 / 0.4) + 0.4 * math.log(0.4 / 0.6)
    assert loss.item() == pytest.approx((first + second) / 2, abs=1e-7)


def test_log_space_forward_kl_is_finite_at_extreme_likelihoods() -> None:
    teacher = torch.tensor([[-1e-12, -10_000.0]], dtype=torch.float32)
    student = torch.tensor([[-10_000.0, -1e-12]], dtype=torch.float32, requires_grad=True)
    mask = torch.ones_like(teacher, dtype=torch.bool)

    loss = alm_forward_kl_loss(
        teacher,
        student,
        mask,
        temperature=1.0,
        epsilon=1e-6,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_no_valid_chunks_returns_differentiable_zero() -> None:
    teacher = torch.tensor([[-0.2, -0.3]])
    student = torch.tensor([[-0.4, -0.5]], requires_grad=True)
    mask = torch.zeros((1, 2), dtype=torch.bool)

    loss = alm_forward_kl_loss(teacher, student, mask)
    loss.backward()

    assert loss.item() == 0.0
    assert student.grad is not None
    assert student.grad.tolist() == [[0.0, 0.0]]
