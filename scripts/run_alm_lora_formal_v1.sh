#!/usr/bin/env bash
set -euo pipefail

: "${AUTODL_ROOT:?set AUTODL_ROOT to the remote workspace root}"

if [[ $# -ne 1 ]] || [[ "$1" != "sft_only" && "$1" != "sft_alm" ]]; then
    echo "usage: $0 {sft_only|sft_alm}" >&2
    exit 2
fi

ARM="$1"
if [[ "${ARM}" == "sft_only" ]]; then
    ALPHA_ALM_VALUE=0.0
else
    ALPHA_ALM_VALUE=1.0
fi

RUN_NAME="alm_lora_formal_v1_authoritative_${ARM}"
LOG_DIR="${AUTODL_ROOT}/logs/${RUN_NAME}"
FORMAL_OUTPUT_DIR="${AUTODL_ROOT}/outputs/${RUN_NAME}"
REPO_DIR=${AUTODL_ROOT}/repo_5251462
MODEL_DIR=${AUTODL_ROOT}/hf-cache/hub/models--Qwen--Qwen2.5-Coder-7B-Instruct/snapshots/c03e6d358207e414f1eca0bb1891e29f1db0e242
TRAIN_DATA=${AUTODL_ROOT}/data/accepted_first_200.jsonl
EXPECTED_DATA_SHA=19014be66857e258925a20e211f853df6f5471e01c110a67364b2250ce5e5e95

mkdir -p "${LOG_DIR}"
if [[ -e "${FORMAL_OUTPUT_DIR}" ]]; then
    echo "refusing to overwrite existing output: ${FORMAL_OUTPUT_DIR}" >&2
    exit 2
fi
if [[ ! -d "${REPO_DIR}" || ! -d "${MODEL_DIR}" || ! -f "${TRAIN_DATA}" ]]; then
    echo "missing repository, model snapshot, or training dataset" >&2
    exit 2
fi
ACTUAL_DATA_SHA="$(sha256sum "${TRAIN_DATA}" | cut -d' ' -f1)"
if [[ "${ACTUAL_DATA_SHA}" != "${EXPECTED_DATA_SHA}" ]]; then
    echo "training dataset SHA-256 mismatch: ${ACTUAL_DATA_SHA}" >&2
    exit 2
fi

exec > >(tee -a "${LOG_DIR}/train.log") 2>&1

GPU_MONITOR_PID=""
finish() {
    status=$?
    if [[ -n "${GPU_MONITOR_PID}" ]]; then
        kill "${GPU_MONITOR_PID}" 2>/dev/null || true
        wait "${GPU_MONITOR_PID}" 2>/dev/null || true
    fi
    printf '%s\n' "${status}" > "${LOG_DIR}/exit_code"
    date --iso-8601=seconds > "${LOG_DIR}/finished_at"
    nvidia-smi \
        --query-gpu=timestamp,name,memory.total,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader > "${LOG_DIR}/gpu_final.csv" || true
}
trap finish EXIT

date --iso-8601=seconds > "${LOG_DIR}/started_at"
printf '%s\n' "52514620a63095b10729fcd98242b571ef2d8b79" > "${LOG_DIR}/code_commit"
printf '%s\n' "${ACTUAL_DATA_SHA}" > "${LOG_DIR}/training_data_sha256"
cp "$0" "${LOG_DIR}/run.sh"
nvidia-smi \
    --query-gpu=timestamp,name,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader > "${LOG_DIR}/gpu_initial.csv"
(
    while true; do
        nvidia-smi \
            --query-gpu=timestamp,memory.used,utilization.gpu \
            --format=csv,noheader,nounits
        sleep 1
    done
) > "${LOG_DIR}/gpu_samples.csv" 2>&1 &
GPU_MONITOR_PID="$!"

source ${AUTODL_ROOT}/activate-topk-distill.sh
cd "${REPO_DIR}"

env \
    CUDA_VISIBLE_DEVICES=0 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH="${REPO_DIR}/src" \
    TRAIN_DATASET="${TRAIN_DATA}" \
    STUDENT_MODEL="${MODEL_DIR}" \
    TRAIN_LIMIT=200 \
    MAX_LENGTH=4096 \
    OUTPUT_DIR="${FORMAL_OUTPUT_DIR}" \
    USE_LORA=1 \
    LORA_R=16 \
    LORA_ALPHA=32 \
    LORA_DROPOUT=0.05 \
    LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    ALPHA_ALM="${ALPHA_ALM_VALUE}" \
    ALM_TEMPERATURE=100.0 \
    ALM_EPSILON=1e-6 \
    LEARNING_RATE=1e-4 \
    EPOCHS=5 \
    MAX_STEPS=-1 \
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
    SAVE_STRATEGY=epoch \
    SAVE_TOTAL_LIMIT=3 \
    RUN_NAME="${RUN_NAME}" \
    ${AUTODL_ROOT}/envs/topk-distill/bin/python \
    examples/train_offline_alm.py
