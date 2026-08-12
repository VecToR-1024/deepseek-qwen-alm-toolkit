#!/usr/bin/env bash
set -euo pipefail

: "${AUTODL_ROOT:?set AUTODL_ROOT to the remote workspace root}"

# Adapted from the completed LiveCodeBench launcher in
# multisource_hard_combined_alpha10_checkpoints_benchmarks_v1_20260804.
RUN_ROOT=${AUTODL_ROOT}/benchmarks/qwen25_7b_instruct_hard_combined_alpha10_compare_v1_20260804
SUITE_ROOT="${RUN_ROOT}/livecodebench"
REPO=${AUTODL_ROOT}/repo_qwen25_instruct_20260804
ASSET_ROOT="${REPO}/runs/qwen25_7b_instruct_hard_combined_alpha10_v1_20260804"
HARNESS=${AUTODL_ROOT}/benchmarks/merged_v3_lora_655_full_v1_20260729
LCB_ROOT=${AUTODL_ROOT}/benchmarks/qwen25coder7b_livecodebench_v1
LCB_SOURCE=${AUTODL_ROOT}/benchmarks/livecodebench-src
SUMMARY_ROOT="${REPO}/runs/multisource_hard_combined_alpha10_checkpoints_benchmarks_v1_20260804"
PY=${AUTODL_ROOT}/envs/topk-distill/bin/python
GPU_MONITOR_PID=""

mkdir -p "${SUITE_ROOT}/provenance" "${RUN_ROOT}/stages"
exec 8>"${SUITE_ROOT}/launcher.lock"
if ! flock -n 8; then
    echo "another LiveCodeBench launcher holds ${SUITE_ROOT}/launcher.lock" >&2
    exit 75
fi

finish_launcher() {
    status=$?
    if [[ -n "${GPU_MONITOR_PID}" ]]; then
        kill "${GPU_MONITOR_PID}" 2>/dev/null || true
        wait "${GPU_MONITOR_PID}" 2>/dev/null || true
    fi
    printf '%s\n' "${status}" > "${SUITE_ROOT}/launcher.exit_code"
    date --iso-8601=seconds > "${SUITE_ROOT}/launcher.finished_at"
    printf '%s\t%s\n' "$(date --iso-8601=seconds)" "${status}" \
        >> "${SUITE_ROOT}/launcher.attempts.tsv"
}
trap finish_launcher EXIT

test -f "${RUN_ROOT}/candidate_plan.json"
test -f "${RUN_ROOT}/livecodebench_strict_manifest.json"
test -f "${RUN_ROOT}/livecodebench_compatible_manifest.json"
test -f "${ASSET_ROOT}/run_lcb_base.py"
test -x "${ASSET_ROOT}/run_lcb_base_score.sh"
test -f "${HARNESS}/run_lcb_adapter.py"
test -x "${HARNESS}/run_lcb_score.sh"
test -f "${REPO}/scripts/recover_lcb_format.py"
test -f "${SUMMARY_ROOT}/summarize_livecodebench_comparison.py"
test "$(git -C "${LCB_SOURCE}" rev-parse HEAD)" = \
    28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24
id evalplus-runner >/dev/null

printf '%s  %s\n' \
    5f00c8d1883b130c4b3762a6ec42ac4349c057fae9b0a700a57a4bd20e4cfd58 \
    "${ASSET_ROOT}/run_lcb_base.py" \
    8fc184353085f2bc5e594850fa05898da71134a07a591b81eff7b527be73b56e \
    "${ASSET_ROOT}/run_lcb_base_score.sh" \
    847343ea570a1382a534b03eb9db14ab4dfe284c3df50e57355784d495eadff2 \
    "${HARNESS}/run_lcb_adapter.py" \
    4f7e4800cabe773300c1ac084f3d8f2ce4568a643411804f40221ff7b82cb928 \
    "${HARNESS}/run_lcb_score.sh" \
    37f5c66c7248fb483961c83a0b00271a6bd9cfab5d5800a785adb7bc824637c9 \
    "${REPO}/scripts/recover_lcb_format.py" \
    | sha256sum --check --status

date --iso-8601=seconds > "${SUITE_ROOT}/launcher.started_at"
nvidia-smi -q > "${SUITE_ROOT}/provenance/nvidia_smi_initial.txt"
(
    while true; do
        nvidia-smi \
            --query-gpu=timestamp,memory.used,utilization.gpu \
            --format=csv,noheader,nounits
        sleep 2
    done
) >> "${SUITE_ROOT}/provenance/gpu_samples.csv" 2>&1 &
GPU_MONITOR_PID="$!"

run_stage() {
    local candidate="$1"
    local stage="$2"
    shift 2
    local stage_root="${RUN_ROOT}/stages/${candidate}"
    local completed="${stage_root}/${stage}.completed"
    local attempt log status
    mkdir -p "${stage_root}"
    if [[ -f "${completed}" ]]; then
        echo "SKIP completed ${candidate}/${stage}"
        return 0
    fi
    attempt="$(date +%Y%m%dT%H%M%S%z)"
    log="${stage_root}/${stage}.${attempt}.log"
    printf '%s\n' "${log}" > "${stage_root}/${stage}.latest_log"
    printf 'running\n' > "${stage_root}/${stage}.status"
    date --iso-8601=seconds > "${stage_root}/${stage}.started_at"
    set +e
    "$@" 2>&1 | tee "${log}"
    status="${PIPESTATUS[0]}"
    set -e
    if grep -Eqi \
        'Traceback \(most recent call last\)|CUDA out of memory|No space left on device|AF_UNIX path too long' \
        "${log}"; then
        status=97
    fi
    printf '%s\n' "${status}" > "${stage_root}/${stage}.exit_code"
    date --iso-8601=seconds > "${stage_root}/${stage}.finished_at"
    printf '%s\t%s\t%s\n' "${attempt}" "${status}" "${log}" \
        >> "${stage_root}/${stage}.attempts.tsv"
    if [[ "${status}" -ne 0 ]]; then
        printf 'failed:%s\n' "${status}" > "${stage_root}/${stage}.status"
        return "${status}"
    fi
    printf 'completed\n' > "${stage_root}/${stage}.status"
    date --iso-8601=seconds > "${completed}"
}

mapfile -t candidate_rows < <(
    "${PY}" - "${RUN_ROOT}/candidate_plan.json" <<'PY'
import json
import sys

plan = json.load(open(sys.argv[1], encoding="utf-8"))
for name in plan["candidate_order"]:
    item = plan["candidates"][name]
    print("\t".join((name, item["role"], item["path"], item.get("adapter_sha256", "-"))))
PY
)

for row in "${candidate_rows[@]}"; do
    IFS=$'\t' read -r candidate role model_path model_sha <<<"${row}"
    strict_root="${SUITE_ROOT}/candidates/${candidate}/strict"
    compatible_root="${SUITE_ROOT}/candidates/${candidate}/compatible"

    if [[ "${role}" == base_model ]]; then
        run_stage "${candidate}" livecodebench_generate \
            env -u LCB_LIMIT LCB_BATCH_SIZE=4 \
            PYTHONPATH="${LCB_SOURCE}:${LCB_ROOT}/pydeps" \
            HF_HOME=${AUTODL_ROOT}/hf-cache HF_HUB_OFFLINE=1 \
            TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
            "${PY}" "${ASSET_ROOT}/run_lcb_base.py" generate \
            --candidate "${candidate}" --base-model "${model_path}" \
            --model-id Qwen/Qwen2.5-7B-Instruct \
            --model-revision a09a35458c702b33eeacc393d103063234e8bc28 \
            --run-root "${strict_root}"
        run_stage "${candidate}" livecodebench_strict_score \
            env -u LCB_LIMIT "${ASSET_ROOT}/run_lcb_base_score.sh" \
            "${candidate}_strict" "${model_path}" \
            Qwen/Qwen2.5-7B-Instruct \
            a09a35458c702b33eeacc393d103063234e8bc28 "${strict_root}"
    else
        run_stage "${candidate}" livecodebench_generate \
            env -u LCB_LIMIT LCB_BATCH_SIZE=4 \
            PYTHONPATH="${LCB_SOURCE}:${LCB_ROOT}/pydeps" \
            HF_HOME=${AUTODL_ROOT}/hf-cache HF_HUB_OFFLINE=1 \
            TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
            "${PY}" "${HARNESS}/run_lcb_adapter.py" generate \
            --candidate "${candidate}" --adapter "${model_path}" \
            --adapter-sha256 "${model_sha}" --run-root "${strict_root}"
        run_stage "${candidate}" livecodebench_strict_score \
            env -u LCB_LIMIT "${HARNESS}/run_lcb_score.sh" \
            "${candidate}_strict" "${model_path}" "${model_sha}" "${strict_root}"
    fi

    mkdir -p "${compatible_root}/results"
    run_stage "${candidate}" livecodebench_recover \
        "${PY}" "${REPO}/scripts/recover_lcb_format.py" \
        --strict-results "${strict_root}/results/codegeneration_1_0.0.json" \
        --generations "${strict_root}/results/generations.jsonl" \
        --mode interface_wrapper \
        --output-results "${compatible_root}/results/codegeneration_1_0.0.json" \
        --output-audit "${compatible_root}/results/recovery_audit.json" --force
    if [[ "${role}" == base_model ]]; then
        run_stage "${candidate}" livecodebench_compatible_score \
            env -u LCB_LIMIT "${ASSET_ROOT}/run_lcb_base_score.sh" \
            "${candidate}_compatible" "${model_path}" \
            Qwen/Qwen2.5-7B-Instruct \
            a09a35458c702b33eeacc393d103063234e8bc28 "${compatible_root}"
    else
        run_stage "${candidate}" livecodebench_compatible_score \
            env -u LCB_LIMIT "${HARNESS}/run_lcb_score.sh" \
            "${candidate}_compatible" "${model_path}" "${model_sha}" \
            "${compatible_root}"
    fi
done

for mode in strict compatible; do
    run_stage comparison "livecodebench_${mode}_summary" \
        "${PY}" "${SUMMARY_ROOT}/summarize_livecodebench_comparison.py" \
        --manifest "${RUN_ROOT}/livecodebench_${mode}_manifest.json" \
        --output-json "${RUN_ROOT}/livecodebench_${mode}_comparison.json" \
        --output-md "${RUN_ROOT}/livecodebench_${mode}_comparison.md"
done

date --iso-8601=seconds > "${SUITE_ROOT}/all.completed_at"
