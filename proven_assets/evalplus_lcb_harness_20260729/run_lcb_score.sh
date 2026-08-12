#!/usr/bin/env bash
set -euo pipefail

: "${AUTODL_ROOT:?set AUTODL_ROOT to the remote workspace root}"

CANDIDATE="${1:?candidate is required}"
ADAPTER="${2:?adapter is required}"
ADAPTER_SHA256="${3:?adapter sha256 is required}"
CANDIDATE_ROOT="${4:?candidate root is required}"

BENCH_ROOT=${AUTODL_ROOT}/benchmarks/merged_v3_lora_655_full_v1_20260729
BASELINE_ROOT=${AUTODL_ROOT}/benchmarks/qwen25coder7b_livecodebench_v1
PY=${AUTODL_ROOT}/envs/topk-distill/bin/python
RUNNER=evalplus-runner
RUNNER_HOME=${AUTODL_ROOT}/benchmark-runner-home
STATUS="${CANDIDATE_ROOT}/logs/evaluate.status"

on_exit() {
    code=$?
    printf '%s\n' "${code}" > "${CANDIDATE_ROOT}/logs/evaluate.exit_code"
    date --iso-8601=seconds > "${CANDIDATE_ROOT}/logs/evaluate.finished_at"
    if [[ "${code}" -ne 0 ]]; then
        printf 'failed:%s\n' "${code}" > "${STATUS}"
    fi
}
trap on_exit EXIT

test -f "${CANDIDATE_ROOT}/results/codegeneration_1_0.0.json"
mkdir -p "${CANDIDATE_ROOT}/logs" "${RUNNER_HOME}/tmp"
chown -R "${RUNNER}:${RUNNER}" "${CANDIDATE_ROOT}" "${RUNNER_HOME}"

printf 'running\n' > "${STATUS}"
date --iso-8601=seconds > "${CANDIDATE_ROOT}/logs/evaluate.started_at"

# Reuses the pinned project runner under the same non-root seccomp boundary
# as the frozen baseline.
runuser -u "${RUNNER}" -- env -i \
    HOME="${RUNNER_HOME}" \
    TMPDIR="${RUNNER_HOME}/tmp" \
    PATH=/usr/bin:/bin:${AUTODL_ROOT}/envs/topk-distill/bin \
    PYTHONPATH=${AUTODL_ROOT}/benchmarks/livecodebench-src:${AUTODL_ROOT}/benchmarks/qwen25coder7b_livecodebench_v1/pydeps \
    USER="${RUNNER}" \
    TOKENIZERS_PARALLELISM=false \
    LCB_LIMIT="${LCB_LIMIT:-}" \
    "${PY}" \
    ${AUTODL_ROOT}/benchmarks/qwen25coder7b_evalplus_v1/seccomp_exec.py \
    "${PY}" "${BENCH_ROOT}/run_lcb_adapter.py" evaluate \
    --candidate "${CANDIDATE}" \
    --adapter "${ADAPTER}" \
    --adapter-sha256 "${ADAPTER_SHA256}" \
    --run-root "${CANDIDATE_ROOT}"

printf 'completed\n' > "${STATUS}"
