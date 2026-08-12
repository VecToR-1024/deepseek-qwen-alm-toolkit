#!/usr/bin/env bash
set -euo pipefail

: "${AUTODL_ROOT:?set AUTODL_ROOT to the remote workspace root}"

# Adapted from the completed HumanEval launcher in
# multisource_hard_combined_alpha10_checkpoints_benchmarks_v1_20260804.
RUN_ROOT=${AUTODL_ROOT}/benchmarks/qwen25_7b_instruct_hard_combined_alpha10_compare_v1_20260804
SUITE_ROOT="${RUN_ROOT}/humaneval"
REPO=${AUTODL_ROOT}/repo_qwen25_instruct_20260804
ASSET_ROOT="${REPO}/runs/qwen25_7b_instruct_hard_combined_alpha10_v1_20260804"
HARNESS=${AUTODL_ROOT}/benchmarks/merged_v3_lora_655_full_v1_20260729
EVALPLUS_ROOT=${AUTODL_ROOT}/benchmarks/qwen25coder7b_evalplus_v1
EVALPLUS_SOURCE=${AUTODL_ROOT}/benchmarks/evalplus-src
SUMMARY_ROOT="${REPO}/runs/multisource_hard_combined_alpha10_checkpoints_benchmarks_v1_20260804"
PY=${AUTODL_ROOT}/envs/topk-distill/bin/python
GPU_MONITOR_PID=""

mkdir -p "${SUITE_ROOT}/provenance" "${RUN_ROOT}/stages"
exec 8>"${SUITE_ROOT}/launcher.lock"
if ! flock -n 8; then
    echo "another HumanEval launcher holds ${SUITE_ROOT}/launcher.lock" >&2
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
test -f "${RUN_ROOT}/humaneval_manifest.json"
test -f "${ASSET_ROOT}/run_evalplus_base.py"
test -f "${HARNESS}/run_evalplus_codegen.py"
test -x "${HARNESS}/run_evalplus_score.sh"
test -f "${SUMMARY_ROOT}/summarize_humaneval_comparison.py"
test "$(git -C "${EVALPLUS_SOURCE}" rev-parse HEAD)" = \
    26d6d00bb1fd0fa37f39c99d5290da67891d1c5e
id evalplus-runner >/dev/null

printf '%s  %s\n' \
    2ffb62f79b1811f74b2a275a826b19e988531a41a1202a85ee26b68d0ce6d284 \
    "${ASSET_ROOT}/run_evalplus_base.py" \
    4de5f0f8e89000da85275361bc0b0091066be30eb1e904ebe19cfff24cb9e448 \
    "${HARNESS}/run_evalplus_codegen.py" \
    a7f5f4ca3e86e07939a628281e210ba5d37a5a8ab9fd507e8b0222d8a4ba3a48 \
    "${HARNESS}/run_evalplus_score.sh" \
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
    candidate_root="${SUITE_ROOT}/candidates/${candidate}/evalplus"
    if [[ "${role}" == base_model ]]; then
        run_stage "${candidate}" humaneval_codegen \
            env PYTHONPATH="${EVALPLUS_SOURCE}:${EVALPLUS_ROOT}/pydeps" \
            XDG_CACHE_HOME="${EVALPLUS_ROOT}/cache" \
            HF_HOME=${AUTODL_ROOT}/hf-cache HF_HUB_OFFLINE=1 \
            TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
            "${PY}" "${ASSET_ROOT}/run_evalplus_base.py" \
            --candidate "${candidate}" --base-model "${model_path}" \
            --model-id Qwen/Qwen2.5-7B-Instruct \
            --model-revision a09a35458c702b33eeacc393d103063234e8bc28 \
            --dataset humaneval --run-root "${candidate_root}" \
            --allow-empty-sanitized
    else
        run_stage "${candidate}" humaneval_codegen \
            env PYTHONPATH="${EVALPLUS_SOURCE}:${EVALPLUS_ROOT}/pydeps" \
            XDG_CACHE_HOME="${EVALPLUS_ROOT}/cache" \
            HF_HOME=${AUTODL_ROOT}/hf-cache HF_HUB_OFFLINE=1 \
            TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
            "${PY}" "${HARNESS}/run_evalplus_codegen.py" \
            --candidate "${candidate}" --adapter "${model_path}" \
            --adapter-sha256 "${model_sha}" --dataset humaneval \
            --run-root "${candidate_root}"
    fi
    run_stage "${candidate}" humaneval_score \
        "${HARNESS}/run_evalplus_score.sh" \
        "${candidate}" "${candidate_root}" humaneval
done

run_stage comparison humaneval_summary \
    env PYTHONPATH="${EVALPLUS_SOURCE}:${EVALPLUS_ROOT}/pydeps" \
    XDG_CACHE_HOME="${EVALPLUS_ROOT}/cache" HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    "${PY}" "${SUMMARY_ROOT}/summarize_humaneval_comparison.py" \
    --manifest "${RUN_ROOT}/humaneval_manifest.json" \
    --output-json "${RUN_ROOT}/humaneval_comparison.json" \
    --output-md "${RUN_ROOT}/humaneval_comparison.md" --force

date --iso-8601=seconds > "${SUITE_ROOT}/all.completed_at"
