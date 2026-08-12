#!/usr/bin/env bash
set -euo pipefail

: "${AUTODL_ROOT:?set AUTODL_ROOT to the remote workspace root}"

RUN_ROOT=${AUTODL_ROOT}/benchmarks/merged_v3_lora_655_full_v1_20260729
TRAIN_ROOT=${AUTODL_ROOT}/experiments/merged_v3_lora_655_v1_20260729
EVALPLUS_ROOT=${AUTODL_ROOT}/benchmarks/qwen25coder7b_evalplus_v1
LCB_ROOT=${AUTODL_ROOT}/benchmarks/qwen25coder7b_livecodebench_v1
PY=${AUTODL_ROOT}/envs/topk-distill/bin/python
MANIFEST="${RUN_ROOT}/benchmark_manifest.json"
GPU_MONITOR_PID=""

finish_launcher() {
    status=$?
    if [[ -n "${GPU_MONITOR_PID}" ]]; then
        kill "${GPU_MONITOR_PID}" 2>/dev/null || true
        wait "${GPU_MONITOR_PID}" 2>/dev/null || true
    fi
    printf '%s\n' "${status}" > "${RUN_ROOT}/launcher.exit_code"
    date --iso-8601=seconds > "${RUN_ROOT}/launcher.finished_at"
    nvidia-smi \
        --query-gpu=timestamp,name,memory.total,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader > "${RUN_ROOT}/gpu_final.csv" || true
}
trap finish_launcher EXIT

test -f "${MANIFEST}"
test -f "${TRAIN_ROOT}/training_audit.json"
"${PY}" -c \
    'import json,sys; assert json.load(open(sys.argv[1]))["status"] == "passed"' \
    "${TRAIN_ROOT}/training_audit.json"
test "$(sha256sum "${TRAIN_ROOT}/training_audit.json" | cut -d' ' -f1)" = \
    "80f26e7d557b2b36a8e86d88258ce43c51e6e63299adc189fe2515490090a5fe"
test "$(sha256sum "${EVALPLUS_ROOT}/mbpp_plus_heldout_manifest.json" | cut -d' ' -f1)" = \
    "a06e7703647b18f82ba49fa5cc40399cdbc8913c9850972d55be1d37b25def3e"
test "$(sha256sum "${LCB_ROOT}/selected_dataset.jsonl" | cut -d' ' -f1)" = \
    "d7b9d4fb14931533c9b0f0be0577c27a912d4512e65b072899364a450ab5b751"

mkdir -p "${RUN_ROOT}/candidates" "${RUN_ROOT}/provenance"
date --iso-8601=seconds > "${RUN_ROOT}/launcher.started_at"
sha256sum \
    "${RUN_ROOT}/benchmark_manifest.json" \
    "${RUN_ROOT}/launch_full_benchmark.sh" \
    "${RUN_ROOT}/run_evalplus_codegen.py" \
    "${RUN_ROOT}/run_evalplus_score.sh" \
    "${RUN_ROOT}/summarize_evalplus_score.py" \
    "${RUN_ROOT}/run_lcb_adapter.py" \
    "${RUN_ROOT}/run_lcb_score.sh" \
    > "${RUN_ROOT}/provenance/runner_sha256.txt"
nvidia-smi -q > "${RUN_ROOT}/provenance/nvidia_smi_initial.txt"
(
    while true; do
        nvidia-smi \
            --query-gpu=timestamp,memory.used,utilization.gpu \
            --format=csv,noheader,nounits
        sleep 2
    done
) > "${RUN_ROOT}/provenance/gpu_samples.csv" 2>&1 &
GPU_MONITOR_PID="$!"

run_stage() {
    local candidate="$1"
    local stage="$2"
    shift 2
    local stage_root="${RUN_ROOT}/candidates/${candidate}/stages"
    local log="${stage_root}/${stage}.log"
    local completed="${stage_root}/${stage}.completed"
    mkdir -p "${stage_root}"
    if [[ -f "${completed}" ]]; then
        echo "SKIP completed ${candidate}/${stage}"
        return 0
    fi
    printf '%s/%s\n' "${candidate}" "${stage}" > "${RUN_ROOT}/current_stage"
    date --iso-8601=seconds > "${stage_root}/${stage}.started_at"
    set +e
    "$@" 2>&1 | tee "${log}"
    status="${PIPESTATUS[0]}"
    set -e
    if grep -E \
        'Traceback \(most recent call last\)|Process SyncManager|AF_UNIX path too long' \
        "${log}" >/dev/null; then
        echo "infrastructure traceback detected in ${candidate}/${stage}" >&2
        status=97
    fi
    printf '%s\n' "${status}" > "${stage_root}/${stage}.exit_code"
    date --iso-8601=seconds > "${stage_root}/${stage}.finished_at"
    if [[ "${status}" -ne 0 ]]; then
        return "${status}"
    fi
    date --iso-8601=seconds > "${completed}"
}

while IFS=$'\t' read -r candidate adapter adapter_sha; do
    candidate_root="${RUN_ROOT}/candidates/${candidate}"
    evalplus_candidate="${candidate_root}/evalplus"
    lcb_candidate="${candidate_root}/livecodebench"
    mkdir -p "${candidate_root}"

    run_stage "${candidate}" "evalplus_humaneval_codegen" \
        env \
        PYTHONPATH=${AUTODL_ROOT}/benchmarks/evalplus-src:${AUTODL_ROOT}/benchmarks/qwen25coder7b_evalplus_v1/pydeps \
        XDG_CACHE_HOME=${AUTODL_ROOT}/benchmarks/qwen25coder7b_evalplus_v1/cache \
        HF_HOME=${AUTODL_ROOT}/hf-cache \
        HF_HUB_OFFLINE=1 \
        TRANSFORMERS_OFFLINE=1 \
        TOKENIZERS_PARALLELISM=false \
        "${PY}" "${RUN_ROOT}/run_evalplus_codegen.py" \
        --candidate "${candidate}" \
        --adapter "${adapter}" \
        --adapter-sha256 "${adapter_sha}" \
        --dataset humaneval \
        --run-root "${evalplus_candidate}"

    run_stage "${candidate}" "evalplus_mbpp_codegen" \
        env \
        PYTHONPATH=${AUTODL_ROOT}/benchmarks/evalplus-src:${AUTODL_ROOT}/benchmarks/qwen25coder7b_evalplus_v1/pydeps \
        XDG_CACHE_HOME=${AUTODL_ROOT}/benchmarks/qwen25coder7b_evalplus_v1/cache \
        HF_HOME=${AUTODL_ROOT}/hf-cache \
        HF_HUB_OFFLINE=1 \
        TRANSFORMERS_OFFLINE=1 \
        TOKENIZERS_PARALLELISM=false \
        "${PY}" "${RUN_ROOT}/run_evalplus_codegen.py" \
        --candidate "${candidate}" \
        --adapter "${adapter}" \
        --adapter-sha256 "${adapter_sha}" \
        --dataset mbpp \
        --run-root "${evalplus_candidate}"

    run_stage "${candidate}" "evalplus_humaneval_score" \
        "${RUN_ROOT}/run_evalplus_score.sh" \
        "${candidate}" "${evalplus_candidate}" humaneval

    run_stage "${candidate}" "evalplus_mbpp_score" \
        "${RUN_ROOT}/run_evalplus_score.sh" \
        "${candidate}" "${evalplus_candidate}" mbpp

    run_stage "${candidate}" "livecodebench_generate" \
        env -u LCB_LIMIT \
        PYTHONPATH=${AUTODL_ROOT}/benchmarks/livecodebench-src:${AUTODL_ROOT}/benchmarks/qwen25coder7b_livecodebench_v1/pydeps \
        HF_HOME=${AUTODL_ROOT}/hf-cache \
        HF_HUB_OFFLINE=1 \
        TRANSFORMERS_OFFLINE=1 \
        TOKENIZERS_PARALLELISM=false \
        LCB_BATCH_SIZE=4 \
        "${PY}" "${RUN_ROOT}/run_lcb_adapter.py" generate \
        --candidate "${candidate}" \
        --adapter "${adapter}" \
        --adapter-sha256 "${adapter_sha}" \
        --run-root "${lcb_candidate}"

    run_stage "${candidate}" "livecodebench_score" \
        env -u LCB_LIMIT \
        "${RUN_ROOT}/run_lcb_score.sh" \
        "${candidate}" "${adapter}" "${adapter_sha}" "${lcb_candidate}"

    date --iso-8601=seconds > "${candidate_root}/completed_at"
done < <(
    "${PY}" -c \
        'import json,sys; m=json.load(open(sys.argv[1])); [(print(name, m["candidates"][name]["adapter"], m["candidates"][name]["adapter_sha256"], sep="\t")) for name in m["candidate_order"]]' \
        "${MANIFEST}"
)

find "${RUN_ROOT}/candidates" -type f -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    > "${RUN_ROOT}/provenance/output_sha256.txt"
date --iso-8601=seconds > "${RUN_ROOT}/all_candidates.completed_at"
