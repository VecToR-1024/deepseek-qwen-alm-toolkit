"""Train Qwen with hard SFT plus offline DeepSeek ALM using optional LoRA."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)

from deepseek_distill.alm_preprocessing import ALMExampleBuilder
from topk_distill.alm_trainer import ALMDataCollator, ALMTrainer


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"set the {name} environment variable")
    return value


def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be a boolean flag")


def load_training_dataset(path: str | Path, limit: int) -> Dataset:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("training limit must be a non-negative integer")
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                source = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"training record line {line_number} is not valid JSON"
                ) from exc
            if not isinstance(source, Mapping):
                raise ValueError(
                    f"training record line {line_number} must be an object"
                )
            request = source.get("request")
            messages = request.get("messages") if isinstance(request, Mapping) else None
            token_rows = source.get("content_tokens")
            if not isinstance(messages, list) or not isinstance(token_rows, list):
                raise ValueError(
                    f"training record line {line_number} is missing trace fields"
                )
            for position, row in enumerate(token_rows):
                if not isinstance(row, Mapping):
                    raise ValueError(
                        f"training record line {line_number} "
                        f"content_tokens[{position}] must be an object"
                    )
            records.append(
                {
                    "schema_version": source.get("schema_version"),
                    "id": source.get("id"),
                    "request": {"messages": messages},
                    "response_text": source.get("response_text"),
                    "content_tokens": [
                        {
                            "bytes": row.get("bytes"),
                            "logprob": row.get("logprob"),
                        }
                        for row in token_rows
                    ],
                }
            )
            if limit and len(records) == limit:
                break
    if limit and len(records) < limit:
        raise ValueError(
            f"training limit is {limit}, but the dataset only contains "
            f"{len(records)} records"
        )
    if not records:
        raise ValueError("training dataset contains no records")
    return Dataset.from_list(records)


def resume_checkpoint_from_env() -> str | bool | None:
    value = os.environ.get("RESUME_FROM_CHECKPOINT")
    if value is None:
        return None
    stripped = value.strip()
    if stripped.lower() in {"", "0", "false", "no", "off"}:
        return None
    if stripped.lower() in {"1", "true", "yes", "on"}:
        return True
    return stripped


def warmup_from_env() -> tuple[float | None, int]:
    warmup_steps_value = os.environ.get("WARMUP_STEPS")
    if warmup_steps_value is not None:
        return None, int(warmup_steps_value)
    return float(os.environ.get("WARMUP_RATIO", "0.0")), 0


def chat_template_kwargs_from_env() -> dict[str, Any]:
    """Parse optional tokenizer chat-template arguments from JSON."""

    raw = os.environ.get("CHAT_TEMPLATE_KWARGS")
    if raw is None or not raw.strip():
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("CHAT_TEMPLATE_KWARGS must be valid JSON") from exc
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise RuntimeError("CHAT_TEMPLATE_KWARGS must be a JSON object")
    return dict(value)


def main() -> None:
    dataset_path = required_env("TRAIN_DATASET")
    model_name = os.environ.get(
        "STUDENT_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct"
    )
    max_length = int(os.environ.get("MAX_LENGTH", "4096"))
    train_limit = int(os.environ.get("TRAIN_LIMIT", "0"))
    seed = int(os.environ.get("SEED", "42"))
    data_seed = int(os.environ.get("DATA_SEED", str(seed)))
    model_revision = os.environ.get("MODEL_REVISION") or None
    chat_template_kwargs = chat_template_kwargs_from_env()
    use_lora = env_flag("USE_LORA", True)
    warmup_ratio, warmup_steps = warmup_from_env()
    set_seed(seed)

    model_source_kwargs: dict[str, Any] = {}
    if model_revision is not None:
        model_source_kwargs["revision"] = model_revision
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        **model_source_kwargs,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    builder = ALMExampleBuilder(
        tokenizer,
        chat_template_kwargs=chat_template_kwargs,
    )
    dataset = load_training_dataset(dataset_path, train_limit)
    source_columns = dataset.column_names
    dataset = dataset.map(
        builder.build,
        remove_columns=source_columns,
        desc="Building offline ALM chunks",
    )
    dataset = dataset.filter(
        lambda row: len(row["input_ids"]) <= max_length
        and any(label != -100 for label in row["labels"]),
        desc="Dropping overlength or empty-SFT examples",
    )
    if len(dataset) == 0:
        raise RuntimeError("no trainable examples remain after ALM preprocessing")

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if use_bf16 else torch.float32,
        **model_source_kwargs,
    )
    if use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        target_modules = os.environ.get(
            "LORA_TARGET_MODULES",
            "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        ).split(",")
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=int(os.environ.get("LORA_R", "16")),
                lora_alpha=int(os.environ.get("LORA_ALPHA", "32")),
                lora_dropout=float(os.environ.get("LORA_DROPOUT", "0.05")),
                target_modules=[
                    module.strip() for module in target_modules if module.strip()
                ],
            ),
        )
        model.print_trainable_parameters()
    model.config.use_cache = False

    output_dir = os.environ.get("OUTPUT_DIR", "outputs/qwen-offline-alm")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=int(os.environ.get("BATCH_SIZE", "1")),
        gradient_accumulation_steps=int(os.environ.get("GRAD_ACCUM", "8")),
        learning_rate=float(os.environ.get("LEARNING_RATE", "2e-4")),
        num_train_epochs=float(os.environ.get("EPOCHS", "1")),
        max_steps=int(os.environ.get("MAX_STEPS", "-1")),
        lr_scheduler_type=os.environ.get("LR_SCHEDULER_TYPE", "linear"),
        warmup_ratio=warmup_ratio,
        warmup_steps=warmup_steps,
        max_grad_norm=float(os.environ.get("MAX_GRAD_NORM", "1.0")),
        gradient_checkpointing=env_flag("GRADIENT_CHECKPOINTING", True),
        bf16=use_bf16,
        fp16=torch.cuda.is_available() and not use_bf16,
        seed=seed,
        data_seed=data_seed,
        logging_steps=int(os.environ.get("LOGGING_STEPS", "10")),
        logging_first_step=env_flag("LOGGING_FIRST_STEP", False),
        save_strategy=os.environ.get("SAVE_STRATEGY", "steps"),
        save_steps=int(os.environ.get("SAVE_STEPS", "500")),
        save_total_limit=int(os.environ.get("SAVE_TOTAL_LIMIT", "2")),
        remove_unused_columns=False,
        report_to="none",
        run_name=os.environ.get("RUN_NAME"),
    )
    trainer = ALMTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=ALMDataCollator(
            pad_token_id=tokenizer.pad_token_id,
            pad_to_multiple_of=8,
        ),
        processing_class=tokenizer,
        alpha_alm=float(os.environ.get("ALPHA_ALM", "1.0")),
        alm_temperature=float(os.environ.get("ALM_TEMPERATURE", "100.0")),
        alm_epsilon=float(os.environ.get("ALM_EPSILON", "1e-6")),
    )
    resume_from_checkpoint = resume_checkpoint_from_env()
    print(
        json.dumps(
            {
                "event": "offline_alm_training_start",
                "alpha_alm": trainer.alpha_alm,
                "bf16": use_bf16,
                "chat_template_kwargs": chat_template_kwargs,
                "dataset_records": len(dataset),
                "data_seed": data_seed,
                "epochs": training_args.num_train_epochs,
                "gradient_accumulation_steps": (
                    training_args.gradient_accumulation_steps
                ),
                "learning_rate": training_args.learning_rate,
                "lr_scheduler_type": str(training_args.lr_scheduler_type),
                "micro_batch_size": training_args.per_device_train_batch_size,
                "model": model_name,
                "model_revision": model_revision,
                "output_dir": output_dir,
                "resume_from_checkpoint": resume_from_checkpoint,
                "seed": seed,
                "training_mode": "bf16_lora" if use_lora else "bf16_full",
                "warmup_ratio": training_args.warmup_ratio,
                "warmup_steps": training_args.warmup_steps,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
