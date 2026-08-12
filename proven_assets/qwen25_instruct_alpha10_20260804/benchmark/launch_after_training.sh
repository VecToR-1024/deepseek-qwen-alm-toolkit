#!/usr/bin/env bash
set -euo pipefail

: "${AUTODL_ROOT:?set AUTODL_ROOT to the remote workspace root}"

# This is the proven two-suite master layout from
# multisource_hard_combined_alpha10_checkpoints_benchmarks_v1_20260804.
# Model-specific commands live in the suite launchers.
RUN_ROOT=${AUTODL_ROOT}/benchmarks/qwen25_7b_instruct_hard_combined_alpha10_compare_v1_20260804
TRAIN_ROOT=${AUTODL_ROOT}/experiments/qwen25_7b_instruct_hard_combined_alpha10_v1_20260804
REPO=${AUTODL_ROOT}/repo_qwen25_instruct_20260804
ASSET_ROOT="${REPO}/runs/qwen25_7b_instruct_hard_combined_alpha10_v1_20260804"
BENCHMARK_ASSETS="${ASSET_ROOT}/benchmark"
MODEL_REVISION=a09a35458c702b33eeacc393d103063234e8bc28
MODEL_DIR="${AUTODL_ROOT}/hf-cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/${MODEL_REVISION}"
LCB_SELECTED_DATASET="${AUTODL_ROOT}/benchmarks/qwen25coder7b_livecodebench_v1/selected_dataset.jsonl"
PY=${AUTODL_ROOT}/envs/topk-distill/bin/python

mkdir -p "${RUN_ROOT}/stages/suites" "${RUN_ROOT}/provenance"
exec 9>"${RUN_ROOT}/launcher.lock"
if ! flock -n 9; then
    echo "another benchmark launcher holds ${RUN_ROOT}/launcher.lock" >&2
    exit 75
fi

finish_launcher() {
    status=$?
    printf '%s\n' "${status}" > "${RUN_ROOT}/launcher.exit_code"
    date --iso-8601=seconds > "${RUN_ROOT}/launcher.finished_at"
    printf '%s\t%s\n' "$(date --iso-8601=seconds)" "${status}" \
        >> "${RUN_ROOT}/launcher.attempts.tsv"
}
trap finish_launcher EXIT

wait_for_training() {
    while [[ ! -f "${TRAIN_ROOT}/training.completed_at" ]]; do
        for stage in model_snapshot base_benchmark_smoke one_epoch_training; do
            status_file="${TRAIN_ROOT}/stages/${stage}/${stage}.status"
            if [[ -f "${status_file}" ]] && grep -q '^failed:' "${status_file}"; then
                echo "training prerequisite failed: ${stage}: $(<"${status_file}")" >&2
                return 2
            fi
        done
        date --iso-8601=seconds > "${RUN_ROOT}/waiting_for_training.heartbeat"
        sleep 60
    done
    test -f "${TRAIN_ROOT}/logs/formal/completed"
    test "$(<"${TRAIN_ROOT}/logs/formal/exit_code")" = 0
}

run_benchmark() {
    local name="$1"
    local launcher="$2"
    local stage_root="${RUN_ROOT}/stages/suites/${name}"
    local status
    mkdir -p "${stage_root}"
    if [[ -f "${stage_root}/completed" ]]; then
        echo "SKIP completed benchmark ${name}"
        return 0
    fi
    date --iso-8601=seconds > "${stage_root}/started_at"
    set +e
    bash "${launcher}" 2>&1 | tee "${stage_root}/launcher.log"
    status="${PIPESTATUS[0]}"
    set -e
    printf '%s\n' "${status}" > "${stage_root}/exit_code"
    date --iso-8601=seconds > "${stage_root}/finished_at"
    if [[ "${status}" -ne 0 ]]; then
        return "${status}"
    fi
    date --iso-8601=seconds > "${stage_root}/completed"
}

date --iso-8601=seconds > "${RUN_ROOT}/launcher.started_at"
wait_for_training
test -d "${MODEL_DIR}"
test -f "${BENCHMARK_ASSETS}/prepare_comparison.py"
test -x "${BENCHMARK_ASSETS}/launch_humaneval.sh"
test -x "${BENCHMARK_ASSETS}/launch_livecodebench.sh"

# Preserve the already-frozen plan on resume. A fresh deployment creates it
# before either suite starts.
if [[ ! -f "${RUN_ROOT}/candidate_plan.json" ]]; then
    "${PY}" "${BENCHMARK_ASSETS}/prepare_comparison.py" \
        --train-root "${TRAIN_ROOT}" \
        --run-root "${RUN_ROOT}" \
        --model-dir "${MODEL_DIR}" \
        --model-id Qwen/Qwen2.5-7B-Instruct \
        --model-revision "${MODEL_REVISION}" \
        --lcb-selected-dataset "${LCB_SELECTED_DATASET}"
fi

sha256sum \
    "${BENCHMARK_ASSETS}/launch_after_training.sh" \
    "${BENCHMARK_ASSETS}/launch_humaneval.sh" \
    "${BENCHMARK_ASSETS}/launch_livecodebench.sh" \
    "${BENCHMARK_ASSETS}/monitor.sh" \
    "${RUN_ROOT}/candidate_plan.json" \
    "${RUN_ROOT}/humaneval_manifest.json" \
    "${RUN_ROOT}/livecodebench_strict_manifest.json" \
    "${RUN_ROOT}/livecodebench_compatible_manifest.json" \
    > "${RUN_ROOT}/provenance/reused_launcher_sha256.txt"

run_benchmark humaneval "${BENCHMARK_ASSETS}/launch_humaneval.sh"
run_benchmark livecodebench "${BENCHMARK_ASSETS}/launch_livecodebench.sh"

date --iso-8601=seconds > "${RUN_ROOT}/all.completed_at"
