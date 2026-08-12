#!/usr/bin/env bash
set -euo pipefail

: "${AUTODL_ROOT:?set AUTODL_ROOT to the remote workspace root}"

RUN_ROOT=${AUTODL_ROOT}/experiments/qwen25_7b_instruct_hard_combined_alpha10_v1_20260804
REPO_DIR=${AUTODL_ROOT}/repo_qwen25_instruct_20260804
MODEL_REVISION=a09a35458c702b33eeacc393d103063234e8bc28
MODEL_DIR="${AUTODL_ROOT}/hf-cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/${MODEL_REVISION}"
TRAIN_DATA=${AUTODL_ROOT}/data/multisource_hard_combined_v1_20260804/frozen_all/training_records.jsonl
DATASET_MANIFEST=${AUTODL_ROOT}/data/multisource_hard_combined_v1_20260804/frozen_all/dataset_manifest.json
EXPECTED_DATA_SHA=64b28fd6d6b090055684c7470da0bd6e6591d52dd72b8651e54c280b6b3b830f
EXPECTED_RECORDS=2041
PREFLIGHT="${RUN_ROOT}/dataset_preflight.json"
PREFLIGHT_MD="${RUN_ROOT}/dataset_preflight.md"
MANIFEST="${RUN_ROOT}/training_manifest.json"
PY=${AUTODL_ROOT}/envs/topk-distill/bin/python
ALPHA_ALM="10.0"
EPOCHS="1"
RESUME_FROM_CHECKPOINT="0"
USE_LORA="1"
GPU_MONITOR_PID=""

HUMANEVAL=${AUTODL_ROOT}/benchmarks/qwen25coder7b_evalplus_v1/cache/evalplus/HumanEvalPlus-v0.1.10.jsonl
MBPP=${AUTODL_ROOT}/benchmarks/qwen25coder7b_evalplus_v1/cache/evalplus/MbppPlus-v0.2.0.jsonl
LCB=${AUTODL_ROOT}/benchmarks/qwen25coder7b_livecodebench_v1/selected_dataset.jsonl

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/provenance"
test -f "${RUN_ROOT}/base_smoke/all.completed_at"
for path in "${REPO_DIR}" "${REPO_DIR}/.codex_commit" "${MODEL_DIR}" \
    "${TRAIN_DATA}" "${DATASET_MANIFEST}" "${HUMANEVAL}" "${MBPP}" "${LCB}"; do
    test -e "${path}" || { echo "missing required input: ${path}" >&2; exit 2; }
done
printf '%s  %s\n' "${EXPECTED_DATA_SHA}" "${TRAIN_DATA}" \
    | sha256sum --check --status
test "$(wc -l < "${TRAIN_DATA}")" -eq "${EXPECTED_RECORDS}"

source ${AUTODL_ROOT}/activate-topk-distill.sh
cd "${REPO_DIR}"

if [[ ! -e "${PREFLIGHT}" && ! -e "${PREFLIGHT_MD}" ]]; then
    "${PY}" scripts/audit_frozen_training_dataset.py \
        --training-data "${TRAIN_DATA}" \
        --tokenizer "${MODEL_DIR}" \
        --benchmark humaneval="${HUMANEVAL}" \
        --benchmark mbpp="${MBPP}" \
        --benchmark livecodebench="${LCB}" \
        --expected-records "${EXPECTED_RECORDS}" \
        --max-length 4096 \
        --output-json "${PREFLIGHT}" \
        --output-md "${PREFLIGHT_MD}" \
        --local-files-only
elif [[ ! -f "${PREFLIGHT}" || ! -f "${PREFLIGHT_MD}" ]]; then
    echo "partial dataset preflight artifacts; refusing to guess" >&2
    exit 2
fi

"${PY}" - "${PREFLIGHT}" "${EXPECTED_DATA_SHA}" "${MODEL_DIR}" <<'PY'
import json
import sys

path, expected_data_sha, expected_model = sys.argv[1:]
report = json.load(open(path, encoding="utf-8"))
assert report["passed"] and all(report["checks"].values())
assert report["inputs"]["training_data_sha256"] == expected_data_sha
assert report["inputs"]["tokenizer"] == expected_model
PY

if [[ ! -e "${MANIFEST}" ]]; then
    "${PY}" - "${MANIFEST}" "${PREFLIGHT}" "${TRAIN_DATA}" \
        "${DATASET_MANIFEST}" "${REPO_DIR}" "${MODEL_DIR}" "${MODEL_REVISION}" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

manifest_path, preflight_path, train_path, dataset_manifest_path, repo, model, revision = sys.argv[1:]

def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

preflight = json.load(open(preflight_path, encoding="utf-8"))
dataset_manifest = json.load(open(dataset_manifest_path, encoding="utf-8"))
commit = Path(repo, ".codex_commit").read_text(encoding="utf-8").strip()
payload = {
    "schema_version": "offline_alm.training_manifest.v1",
    "run_id": "qwen25_7b_instruct_hard_combined_alpha10_v1_20260804",
    "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "training_started_by_manifest_creation": False,
    "dataset": {
        "path": train_path,
        "records": 2041,
        "sha256": sha256(train_path),
        "bytes": Path(train_path).stat().st_size,
        "dataset_manifest": dataset_manifest_path,
        "dataset_manifest_sha256": sha256(dataset_manifest_path),
        "source_counts": dataset_manifest["source_counts"]["training"],
    },
    "student_preprocessing": {
        "rebuilt_for_new_tokenizer": True,
        "reused_coder_input_ids_or_chunks": False,
        "preflight": preflight_path,
        "preflight_sha256": sha256(preflight_path),
        "checks": preflight["checks"],
    },
    "training_code": {"repo": repo, "commit": commit},
    "initialization": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "model_revision": revision,
        "base_model": model,
        "resume_from_checkpoint": None,
        "adapter": "BF16 LoRA",
        "quantized": False,
    },
    "config": {
        "objective": "hard_sft_loss + 10.0 * alm_loss",
        "alpha_alm": 10.0,
        "epochs": 1,
        "learning_rate": 1e-4,
        "micro_batch_size": 1,
        "gradient_accumulation": 8,
        "max_length": 4096,
        "warmup_steps": 6,
        "scheduler": "linear",
        "seed": 20260727,
        "data_seed": 20260727,
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        },
        "alm_temperature": 100.0,
        "alm_epsilon": 1e-6,
    },
    "selection": {
        "planned_checkpoint": "epoch_1_final",
        "full_benchmark_started_automatically": False,
    },
}
Path(manifest_path).write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
PY
fi

sha256sum "${MANIFEST}" "${PREFLIGHT}" "${TRAIN_DATA}" \
    > "${RUN_ROOT}/provenance/training_inputs.sha256"
"${PY}" -m pip freeze > "${RUN_ROOT}/provenance/pip_freeze.txt"
nvidia-smi -q > "${RUN_ROOT}/provenance/nvidia_smi_initial.txt"

run_training() {
    local phase="$1"
    local max_steps="$2"
    local output_dir log_dir limit epochs save_strategy
    if [[ "${phase}" == smoke ]]; then
        output_dir="${RUN_ROOT}/smoke/alpha10"
        log_dir="${RUN_ROOT}/logs/smoke"
        limit=16
        epochs=1
        save_strategy=no
    else
        output_dir="${RUN_ROOT}/outputs/alpha10"
        log_dir="${RUN_ROOT}/logs/formal"
        limit="${EXPECTED_RECORDS}"
        epochs="${EPOCHS}"
        save_strategy=epoch
    fi
    if [[ -f "${log_dir}/completed" ]]; then
        echo "SKIP completed training phase ${phase}"
        return 0
    fi
    if [[ -e "${output_dir}" || -e "${log_dir}" ]]; then
        echo "partial ${phase} output exists; refusing unsafe resume" >&2
        return 2
    fi
    mkdir -p "${log_dir}"
    printf 'base_model=%s\nresume_from_checkpoint=null\n' "${MODEL_DIR}" \
        > "${log_dir}/initialization_contract.txt"
    date --iso-8601=seconds > "${log_dir}/started_at"
    (
        while true; do
            nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu \
                --format=csv,noheader,nounits
            sleep 1
        done
    ) > "${log_dir}/gpu_samples.csv" 2>&1 &
    GPU_MONITOR_PID="$!"

    set +e
    env \
        CUDA_VISIBLE_DEVICES=0 \
        HF_HUB_OFFLINE=1 \
        TRANSFORMERS_OFFLINE=1 \
        TOKENIZERS_PARALLELISM=false \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        PYTHONPATH="${REPO_DIR}/src" \
        TRAIN_DATASET="${TRAIN_DATA}" \
        STUDENT_MODEL="${MODEL_DIR}" \
        TRAIN_LIMIT="${limit}" \
        MAX_LENGTH=4096 \
        OUTPUT_DIR="${output_dir}" \
        RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT}" \
        USE_LORA="${USE_LORA}" \
        LORA_R=16 \
        LORA_ALPHA=32 \
        LORA_DROPOUT=0.05 \
        LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
        ALPHA_ALM="${ALPHA_ALM}" \
        ALM_TEMPERATURE=100.0 \
        ALM_EPSILON=1e-6 \
        LEARNING_RATE=1e-4 \
        EPOCHS="${epochs}" \
        MAX_STEPS="${max_steps}" \
        BATCH_SIZE=1 \
        GRAD_ACCUM=8 \
        LR_SCHEDULER_TYPE=linear \
        WARMUP_STEPS=6 \
        MAX_GRAD_NORM=1.0 \
        GRADIENT_CHECKPOINTING=1 \
        SEED=20260727 \
        DATA_SEED=20260727 \
        LOGGING_STEPS=1 \
        LOGGING_FIRST_STEP=1 \
        SAVE_STRATEGY="${save_strategy}" \
        SAVE_TOTAL_LIMIT=2 \
        RUN_NAME="qwen25_7b_instruct_hard_combined_alpha10_${phase}" \
        "${PY}" examples/train_offline_alm.py \
        2>&1 | tee "${log_dir}/train.log"
    status="${PIPESTATUS[0]}"
    set -e

    kill "${GPU_MONITOR_PID}" 2>/dev/null || true
    wait "${GPU_MONITOR_PID}" 2>/dev/null || true
    GPU_MONITOR_PID=""
    printf '%s\n' "${status}" > "${log_dir}/exit_code"
    date --iso-8601=seconds > "${log_dir}/finished_at"
    if [[ "${status}" -ne 0 ]]; then
        return "${status}"
    fi
    if grep -Eqi '(^|[^[:alpha:]])(nan|inf)([^[:alpha:]]|$)|out of memory|traceback|fatal' \
        "${log_dir}/train.log"; then
        echo "fatal or non-finite pattern in ${phase} log" >&2
        return 2
    fi
    date --iso-8601=seconds > "${log_dir}/completed"
}

run_training smoke 1
run_training formal -1

test -f "${RUN_ROOT}/outputs/alpha10/checkpoint-256/adapter_model.safetensors"
test -f "${RUN_ROOT}/outputs/alpha10/checkpoint-256/trainer_state.json"
find "${RUN_ROOT}/outputs/alpha10" -maxdepth 2 -type f -print0 \
    | sort -z | xargs -0 sha256sum > "${RUN_ROOT}/logs/formal/output_sha256.txt"
date --iso-8601=seconds > "${RUN_ROOT}/training.completed_at"
