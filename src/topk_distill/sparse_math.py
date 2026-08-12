"""Small reference implementations for checking sparse distillation data."""

from __future__ import annotations

import math
from collections.abc import Sequence


def _tail_logprob(logprobs: Sequence[float]) -> float:
    mass = sum(math.exp(value) for value in logprobs)
    if mass > 1.0 + 1e-7:
        raise ValueError("top-k probability mass cannot be greater than one")
    return math.log(max(1.0 - min(mass, 1.0), 1e-30))


def _renormalize(logprobs: Sequence[float]) -> list[float]:
    normalizer = math.log(sum(math.exp(value) for value in logprobs))
    return [value - normalizer for value in logprobs]


def sparse_forward_kl(
    *,
    teacher_logprobs: Sequence[float],
    student_logprobs: Sequence[float],
    add_tail: bool = True,
) -> float:
    """Compute KL(teacher || student) on one shared top-k support."""
    if not teacher_logprobs or len(teacher_logprobs) != len(student_logprobs):
        raise ValueError("teacher and student logprobs must have the same non-zero length")

    teacher = [float(value) for value in teacher_logprobs]
    student = [float(value) for value in student_logprobs]
    if add_tail:
        teacher.append(_tail_logprob(teacher))
        student.append(_tail_logprob(student))
    else:
        teacher = _renormalize(teacher)
        student = _renormalize(student)

    return sum(math.exp(t_logp) * (t_logp - s_logp) for t_logp, s_logp in zip(teacher, student, strict=True))
