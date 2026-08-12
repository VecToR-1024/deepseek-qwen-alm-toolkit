"""Utilities for pluggable top-k logprob distillation with TRL."""

from .contracts import normalize_sequence_logprobs
from .client import HttpSequenceLogprobClient, SequenceLogprobClient
from .sparse_math import sparse_forward_kl

__all__ = [
    "HttpSequenceLogprobClient",
    "SequenceLogprobClient",
    "normalize_sequence_logprobs",
    "sparse_forward_kl",
]
