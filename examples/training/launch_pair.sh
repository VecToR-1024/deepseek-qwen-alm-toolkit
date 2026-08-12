#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_DATASET="${TRAIN_DATASET:?set TRAIN_DATASET to a frozen normalized JSONL file}"
STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/comparison_v1}"

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
        OUTPUT_DIR="${output_dir}" \
        USE_LORA=1 \
        LORA_R=16 \
        LORA_ALPHA=32 \
        LORA_DROPOUT=0.05 \
        LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
        MAX_LENGTH=4096 \
        ALPHA_ALM="${alpha}" \
        ALM_TEMPERATURE=100.0 \
        ALM_EPSILON=1e-6 \
        LEARNING_RATE=1e-4 \
        EPOCHS=1 \
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
        SAVE_TOTAL_LIMIT=2 \
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
    'training_mode=bf16_lora' \
    'objective_sft=hard_sft_loss' \
    'objective_alm=hard_sft_loss+10.0*alm_loss' \
    > "${OUTPUT_ROOT}/experiment_contract.txt"
run_arm sft_only 0.0
run_arm sft_alm 10.0
date --iso-8601=seconds > "${OUTPUT_ROOT}/all.completed_at"
