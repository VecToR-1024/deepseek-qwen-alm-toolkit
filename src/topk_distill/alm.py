"""PyTorch primitives for Approximate Likelihood Matching (ALM).

These functions adapt the causal actual-token extraction, chunk likelihood
aggregation, and binary divergence from tokenkit's official PyTorch ALM guide:
https://github.com/bminixhofer/tokenkit/blob/main/docs/pytorch_alm_from_scratch.ipynb

The default objective is the exact binary forward KL.  The guide's binary
cross-entropy has the same student gradient and differs only by the fixed
teacher Bernoulli entropy.
"""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def causal_actual_token_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Return log p(x[t] | x[:t]) for token positions 1..L-1.

    The explicit ``logits[:, :-1]`` / ``input_ids[:, 1:]`` shift mirrors the
    official ALM notebook and prevents the common off-by-one distillation bug.
    """
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocabulary]")
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    if logits.shape[:2] != input_ids.shape:
        raise ValueError("logits and input_ids batch/sequence dimensions must match")
    if logits.shape[-1] <= 0:
        raise ValueError("logits vocabulary dimension must be positive")

    # Fused cross-entropy computes target -logprobs without materializing a
    # second [batch, sequence, vocabulary] float32 log-softmax tensor.
    shifted_logits = logits[:, :-1].transpose(1, 2)
    shifted_ids = input_ids[:, 1:]
    return -F.cross_entropy(shifted_logits, shifted_ids, reduction="none").float()


def aggregate_chunk_logprobs(
    actual_token_logprobs: torch.Tensor,
    token_chunk_ids: torch.Tensor,
    *,
    num_chunks: int,
) -> torch.Tensor:
    """Sum token logprobs into chunk log-likelihoods with one scatter pass."""
    if actual_token_logprobs.ndim != 2:
        raise ValueError("actual_token_logprobs must have shape [batch, tokens]")
    if token_chunk_ids.shape != actual_token_logprobs.shape:
        raise ValueError("token_chunk_ids must match actual_token_logprobs shape")
    if token_chunk_ids.dtype == torch.bool or token_chunk_ids.dtype.is_floating_point:
        raise ValueError("token_chunk_ids must use an integer dtype")
    if isinstance(num_chunks, bool) or not isinstance(num_chunks, int) or num_chunks < 0:
        raise ValueError("num_chunks must be a non-negative integer")

    valid = token_chunk_ids >= 0
    if num_chunks == 0:
        if token_chunk_ids.device.type == "cpu" and bool(valid.any()):
            raise ValueError("token_chunk_ids contains a chunk but num_chunks is zero")
        return actual_token_logprobs.new_zeros((actual_token_logprobs.shape[0], 0))
    if token_chunk_ids.device.type == "cpu" and bool(
        (token_chunk_ids[valid] >= num_chunks).any()
    ):
        raise ValueError("token_chunk_ids contains an out-of-range chunk index")

    safe_ids = token_chunk_ids.clamp_min(0)
    values = actual_token_logprobs * valid.to(actual_token_logprobs.dtype)
    chunks = actual_token_logprobs.new_zeros(
        (actual_token_logprobs.shape[0], num_chunks)
    )
    return chunks.scatter_add(dim=1, index=safe_ids, src=values)


def _log1mexp(log_probability: torch.Tensor) -> torch.Tensor:
    """Compute log(1-exp(x)) stably for x <= 0."""
    log_half = -math.log(2.0)
    return torch.where(
        log_probability < log_half,
        torch.log1p(-torch.exp(log_probability)),
        torch.log(-torch.expm1(log_probability)),
    )


def _weighted_log(weight: torch.Tensor, log_value: torch.Tensor) -> torch.Tensor:
    """Evaluate weight * log_value with the convention 0 * -inf = 0."""
    return torch.where(weight == 0, torch.zeros_like(log_value), weight * log_value)


def binary_forward_kl_from_logprobs(
    teacher_logprobs: torch.Tensor,
    student_logprobs: torch.Tensor,
    *,
    temperature: float = 100.0,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Elementwise KL(Ber(q_teacher) || Ber(q_student)) in log space.

    A chunk log-likelihood is the log of a Bernoulli "success" probability.
    Dividing that log-likelihood by ``temperature`` is the binarization used
    by tokenkit.  Subtracting ``epsilon`` keeps ``log(1-p)`` defined near one.
    """
    if teacher_logprobs.shape != student_logprobs.shape:
        raise ValueError("teacher and student chunk logprobs must have the same shape")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and non-negative")

    log_q = teacher_logprobs.detach().float() / temperature - epsilon
    log_p = student_logprobs.float() / temperature - epsilon
    q = torch.exp(log_q)
    one_minus_q = -torch.expm1(log_q)

    teacher_entropy_terms = torch.xlogy(q, q) + torch.xlogy(
        one_minus_q, one_minus_q
    )
    cross_terms = _weighted_log(q, log_p) + _weighted_log(
        one_minus_q, _log1mexp(log_p)
    )
    return teacher_entropy_terms - cross_terms


def alm_forward_kl_loss(
    teacher_chunk_logprobs: torch.Tensor,
    student_chunk_logprobs: torch.Tensor,
    chunk_mask: torch.Tensor,
    *,
    temperature: float = 100.0,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Mean binary forward KL over valid aligned chunks only."""
    if chunk_mask.shape != teacher_chunk_logprobs.shape:
        raise ValueError("chunk_mask must match teacher chunk logprobs")
    if student_chunk_logprobs.shape != teacher_chunk_logprobs.shape:
        raise ValueError("student and teacher chunk logprobs must have the same shape")
    if chunk_mask.dtype != torch.bool:
        raise ValueError("chunk_mask must be boolean")
    elementwise = binary_forward_kl_from_logprobs(
        teacher_chunk_logprobs,
        student_chunk_logprobs,
        temperature=temperature,
        epsilon=epsilon,
    )
    weights = chunk_mask.to(elementwise.dtype)
    return (elementwise * weights).sum() / weights.sum().clamp_min(1.0)
