"""Minimal TRL run using a provider-specific sequence-logprob endpoint."""

from __future__ import annotations

import os

import torch
from datasets import load_dataset
from trl.experimental.distillation import DistillationConfig

from topk_distill.client import HttpSequenceLogprobClient
from topk_distill.trainer import PluggableDistillationTrainer


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"set the {name} environment variable")
    return value


def main() -> None:
    endpoint = required_env("TEACHER_SCORE_URL")
    dataset_path = required_env("TRAIN_DATASET")
    api_key = os.environ.get("TEACHER_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None

    teacher_client = HttpSequenceLogprobClient(
        endpoint,
        headers=headers,
        supports_actual_logprobs=False,
    )
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    config = DistillationConfig(
        output_dir=os.environ.get("OUTPUT_DIR", "outputs/topk-distill"),
        use_teacher_server=False,  # PluggableDistillationTrainer enables its own client.
        loss_top_k=int(os.environ.get("TOP_K", "8")),
        loss_add_tail=True,
        beta=0.0,  # Forward KL only needs the teacher's own top-k support.
        lmbda=float(os.environ.get("ON_POLICY_RATIO", "1.0")),
        max_length=int(os.environ.get("MAX_LENGTH", "1024")),
        max_completion_length=int(os.environ.get("MAX_COMPLETION_LENGTH", "512")),
        per_device_train_batch_size=int(os.environ.get("BATCH_SIZE", "1")),
        gradient_accumulation_steps=int(os.environ.get("GRAD_ACCUM", "8")),
        learning_rate=float(os.environ.get("LEARNING_RATE", "1e-6")),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        report_to="none",
    )
    trainer = PluggableDistillationTrainer(
        model=os.environ.get("STUDENT_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
        args=config,
        train_dataset=dataset,
        teacher_client=teacher_client,
    )
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
