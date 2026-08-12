#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_DATASET="${TRAIN_DATASET:?set TRAIN_DATASET to a frozen normalized JSONL file}"
STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen3-0.6B}"
MODEL_REVISION="${MODEL_REVISION:-c1899de289a04d12100db370d81485cdf75e47ca}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/qwen3_0_6b_full_pair_v1}"
read -r DATASET_SHA256 _ < <(sha256sum "${TRAIN_DATASET}")
DATASET_RECORDS="$(wc -l < "${TRAIN_DATASET}")"
TOOLKIT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || printf 'unavailable')"

run_arm() {
    local name="$1"
    local alpha="$2"
    local arm_root="${OUTPUT_ROOT}/${name}"
    local output_dir="${arm_root}/model"
    local log="${arm_root}/train.log"
    if [[ -f "${arm_root}/completed" ]]; then
        echo "SKIP completed ${name}"
        return 0
    fi
    if [[ -e "${output_dir}" ]]; then
        echo "refusing partial output without an explicit resume plan: ${output_dir}" >&2
        return 2
    fi
    mkdir -p "${arm_root}"
    date --iso-8601=seconds > "${arm_root}/started_at"
    set +e
    env \
        PYTHONPATH="${REPO_ROOT}/src" \
        TRAIN_DATASET="${TRAIN_DATASET}" \
        STUDENT_MODEL="${STUDENT_MODEL}" \
        MODEL_REVISION="${MODEL_REVISION}" \
        CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}' \
        OUTPUT_DIR="${output_dir}" \
        TRAIN_LIMIT="${TRAIN_LIMIT:-0}" \
        USE_LORA=0 \
        MAX_LENGTH="${MAX_LENGTH:-4096}" \
        ALPHA_ALM="${alpha}" \
        ALM_TEMPERATURE="${ALM_TEMPERATURE:-100.0}" \
        ALM_EPSILON="${ALM_EPSILON:-1e-6}" \
        LEARNING_RATE="${LEARNING_RATE:-2e-5}" \
        EPOCHS="${EPOCHS:-1}" \
        MAX_STEPS="${MAX_STEPS:--1}" \
        BATCH_SIZE="${BATCH_SIZE:-4}" \
        GRAD_ACCUM="${GRAD_ACCUM:-4}" \
        LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-linear}" \
        WARMUP_RATIO="${WARMUP_RATIO:-0.03}" \
        MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}" \
        GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}" \
        SEED="${SEED:-20260812}" \
        DATA_SEED="${DATA_SEED:-20260812}" \
        LOGGING_STEPS="${LOGGING_STEPS:-1}" \
        LOGGING_FIRST_STEP=1 \
        SAVE_STRATEGY="${SAVE_STRATEGY:-epoch}" \
        SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}" \
        RESUME_FROM_CHECKPOINT=0 \
        RUN_NAME="${name}" \
        "${PYTHON_BIN}" "${REPO_ROOT}/examples/train_offline_alm.py" \
        2>&1 | tee "${log}"
    status="${PIPESTATUS[0]}"
    set -e
    printf '%s\n' "${status}" > "${arm_root}/exit_code"
    date --iso-8601=seconds > "${arm_root}/finished_at"
    if [[ "${status}" -ne 0 ]]; then
        return "${status}"
    fi
    if grep -Eqi 'out of memory|traceback|(^|[^[:alpha:]])(nan|inf)([^[:alpha:]]|$)' "${log}"; then
        echo "fatal or non-finite pattern detected in ${name}" >&2
        return 2
    fi
    date --iso-8601=seconds > "${arm_root}/completed"
}

mkdir -p "${OUTPUT_ROOT}"
printf '%s\n' \
    'schema_version=offline_alm.experiment_contract.v1' \
    'validation_status=data_contract_only' \
    'training_mode=bf16_full' \
    "toolkit_commit=${TOOLKIT_COMMIT}" \
    "training_dataset=${TRAIN_DATASET}" \
    "dataset_sha256=${DATASET_SHA256}" \
    "dataset_records=${DATASET_RECORDS}" \
    "student_model=${STUDENT_MODEL}" \
    "model_revision=${MODEL_REVISION}" \
    'chat_template_kwargs={"enable_thinking": false}' \
    'objective_sft=hard_sft_loss' \
    'objective_alm=hard_sft_loss+10.0*alm_loss' \
    "max_length=${MAX_LENGTH:-4096}" \
    "learning_rate=${LEARNING_RATE:-2e-5}" \
    "epochs=${EPOCHS:-1}" \
    "micro_batch_size=${BATCH_SIZE:-4}" \
    "gradient_accumulation_steps=${GRAD_ACCUM:-4}" \
    "lr_scheduler_type=${LR_SCHEDULER_TYPE:-linear}" \
    "warmup_ratio=${WARMUP_RATIO:-0.03}" \
    "seed=${SEED:-20260812}" \
    "data_seed=${DATA_SEED:-20260812}" \
    > "${OUTPUT_ROOT}/experiment_contract.txt"
run_arm sft_only 0.0
run_arm sft_alm 10.0
date --iso-8601=seconds > "${OUTPUT_ROOT}/all.completed_at"
