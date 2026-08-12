# Qwen3-0.6B full-finetune runbook

## Status and boundary

The Qwen3 tokenizer/data contract is validated. The full-parameter BF16
optimization settings and launcher are not yet GPU-proven, so begin with a
bounded smoke run. This path keeps the existing PyTorch/Transformers/TRL ALM
stack and does not change the ALM objective.

The required template setting is:

```text
CHAT_TEMPLATE_KWARGS={"enable_thinking": false}
```

It must be identical for prompt rendering, teacher-forced completion rendering,
preflight auditing, training, and benchmark generation.

## 1. Freeze and audit the exact input

Do not train directly from a mutable accepted queue. Freeze one JSONL, record
its count and SHA-256, and run the hard gates. For Qwen3 include the template
argument explicitly:

```bash
python scripts/audit_frozen_training_dataset.py \
  --training-data /data/frozen/training_records.jsonl \
  --tokenizer Qwen/Qwen3-0.6B \
  --chat-template-kwargs '{"enable_thinking": false}' \
  --expected-records 3000 \
  --max-length 4096 \
  --benchmark humaneval=/data/benchmarks/humaneval.jsonl \
  --benchmark mbpp=/data/benchmarks/mbppplus.jsonl \
  --benchmark lcb=/data/benchmarks/lcb.jsonl \
  --output-json /data/frozen/preflight.json \
  --output-md /data/frozen/preflight.md
```

Replace `3000` with the exact frozen record count. Do not weaken a failed gate
to make the run start.

## 2. Run a two-step GPU smoke

On the target 48GB GPU, use a new output directory:

```bash
export TRAIN_DATASET=/data/frozen/training_records.jsonl
export OUTPUT_ROOT=/data/runs/qwen3_0_6b_full_smoke_v1
export TRAIN_LIMIT=8
export MAX_STEPS=2
bash examples/training/launch_qwen3_0_6b_full_pair.sh
bash examples/training/monitor_training.sh "$OUTPUT_ROOT"
```

Accept the smoke only if both arms complete, losses are finite, GPU memory is
stable, the saved models/tokenizers reload, and the log announces all of:

- `training_mode=bf16_full`
- the pinned model revision
- `chat_template_kwargs={"enable_thinking": false}`
- `alpha_alm=0.0` for SFT-only and `alpha_alm=10.0` for SFT+ALM

## 3. Launch the real pair from scratch

Remove `TRAIN_LIMIT` and `MAX_STEPS` from the environment and choose another
empty output directory. The launcher intentionally refuses a partial model
directory; resume must be an explicit, separately audited decision.

```bash
unset TRAIN_LIMIT MAX_STEPS
export OUTPUT_ROOT=/data/runs/qwen3_0_6b_full_pair_v1
nohup bash examples/training/launch_qwen3_0_6b_full_pair.sh \
  > "$OUTPUT_ROOT.launcher.log" 2>&1 &
bash examples/training/monitor_training.sh "$OUTPUT_ROOT"
```

The candidate defaults are one epoch, learning rate `2e-5`, micro-batch four,
gradient accumulation four, linear decay, and 3% warmup. These are starting
values, not benchmark-proven hyperparameters. Keep the frozen data, seed,
decoding contract, and benchmark harness fixed when comparing the two arms.

## 4. Promote assets only after validation

After the run, archive the exact launcher/config, environment package list,
GPU/driver information, frozen-data manifest, logs, checkpoints selected for
evaluation, benchmark commands, and result hashes. Only then copy a sanitized
snapshot into `proven_assets/`.
