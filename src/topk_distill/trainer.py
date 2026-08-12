"""Optional TRL DistillationTrainer with an injected top-k teacher client."""

from __future__ import annotations

from typing import Any

from trl.experimental.distillation import DistillationTrainer

from .client import SequenceLogprobClient


class PluggableDistillationTrainer(DistillationTrainer):
    """Use any ``SequenceLogprobClient`` instead of TRL's built-in vLLM client.

    Configure ``DistillationConfig(use_teacher_server=False)``. This class
    enables the server path only after the base trainer has initialized, which
    avoids constructing TRL's vLLM-specific client and calling ``/health/``.
    """

    def __init__(
        self,
        *args: Any,
        teacher_client: SequenceLogprobClient,
        **kwargs: Any,
    ) -> None:
        config = kwargs.get("args")
        if config is None:
            raise ValueError("args must contain a DistillationConfig")
        if config.use_teacher_server:
            raise ValueError("set use_teacher_server=False when injecting a custom teacher_client")
        if config.loss_top_k <= 0:
            raise ValueError("loss_top_k must be positive for a remote teacher client")
        if config.beta > 0 and not teacher_client.supports_actual_logprobs:
            raise ValueError("beta > 0 requires actual-token logprobs from the teacher client")
        if config.beta > 0 and config.loss_top_k != 1:
            raise ValueError("TRL server-backed reverse KL/JSD currently requires loss_top_k=1")

        super().__init__(*args, **kwargs)
        self.teacher_client = teacher_client
        self.use_teacher_server = True
        self.args.use_teacher_server = True
