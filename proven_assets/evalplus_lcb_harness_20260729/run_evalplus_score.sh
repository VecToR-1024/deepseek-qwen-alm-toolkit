#!/usr/bin/env bash
set -euo pipefail

: "${AUTODL_ROOT:?set AUTODL_ROOT to the remote workspace root}"

CANDIDATE="${1:?candidate is required}"
CANDIDATE_ROOT="${2:?candidate root is required}"
DATASET="${3:?dataset is required}"
case "${DATASET}" in
    humaneval|mbpp) ;;
    *) echo "unsupported dataset: ${DATASET}" >&2; exit 2 ;;
esac

BENCH_ROOT=${AUTODL_ROOT}/benchmarks/merged_v3_lora_655_full_v1_20260729
BASELINE_ROOT=${AUTODL_ROOT}/benchmarks/qwen25coder7b_evalplus_v1
DEPS="${BASELINE_ROOT}/pydeps"
PY=${AUTODL_ROOT}/envs/topk-distill/bin/python
RUNNER=evalplus-runner
RESULT_DIR="${CANDIDATE_ROOT}/results/${DATASET}"
SAMPLE="${RESULT_DIR}/model_hf_temp_0.0.jsonl"
OUTPUT="${RESULT_DIR}/model_hf_temp_0.0.eval_results.json"
STATUS="${CANDIDATE_ROOT}/logs/${DATASET}_score.status"
RUNNER_HOME=${AUTODL_ROOT}/benchmark-runner-home

on_exit() {
    code=$?
    printf '%s\n' "${code}" > "${CANDIDATE_ROOT}/logs/${DATASET}_score.exit_code"
    date --iso-8601=seconds > "${CANDIDATE_ROOT}/logs/${DATASET}_score.finished_at"
    if [[ "${code}" -ne 0 ]]; then
        printf 'failed:%s\n' "${code}" > "${STATUS}"
    fi
}
trap on_exit EXIT

test -f "${SAMPLE}"
test "$(readlink -m "${RESULT_DIR}")" = "${CANDIDATE_ROOT}/results/${DATASET}"
test "$(readlink -m "${BASELINE_ROOT}/cache/evalplus")" = \
    "${BASELINE_ROOT}/cache/evalplus"
mkdir -p "${CANDIDATE_ROOT}/logs" "${RUNNER_HOME}/tmp"
chown -R "${RUNNER}:${RUNNER}" "${RESULT_DIR}" "${BASELINE_ROOT}/cache/evalplus"
chown -R "${RUNNER}:${RUNNER}" "${RUNNER_HOME}"

printf 'running\n' > "${STATUS}"
date --iso-8601=seconds > "${CANDIDATE_ROOT}/logs/${DATASET}_score.started_at"

# Pinned official evaluator:
# https://github.com/evalplus/evalplus/blob/26d6d00bb1fd0fa37f39c99d5290da67891d1c5e/evalplus/evaluate.py
runuser -u "${RUNNER}" -- env -i \
    HOME="${RUNNER_HOME}" \
    TMPDIR="${RUNNER_HOME}/tmp" \
    PATH=/usr/bin:/bin:${AUTODL_ROOT}/envs/topk-distill/bin:"${DEPS}/bin" \
    PYTHONPATH="${DEPS}" \
    XDG_CACHE_HOME="${BASELINE_ROOT}/cache" \
    TOKENIZERS_PARALLELISM=false \
    "${PY}" "${BASELINE_ROOT}/seccomp_exec.py" \
    "${DEPS}/bin/evalplus.evaluate" "${DATASET}" \
    --samples="${SAMPLE}" \
    --parallel=10 \
    --base_only=False \
    --test_details=False \
    --min_time_limit=4.0 \
    --gt_time_limit_factor=4.0 \
    --output_file="${OUTPUT}"

summary_args=(
    --candidate "${CANDIDATE}"
    --dataset "${DATASET}"
    --candidate-root "${CANDIDATE_ROOT}"
)
if [[ "${DATASET}" == "mbpp" ]]; then
    summary_args+=(
        --heldout-manifest
        "${BASELINE_ROOT}/mbpp_plus_heldout_manifest.json"
    )
fi
"${PY}" "${BENCH_ROOT}/summarize_evalplus_score.py" "${summary_args[@]}"
printf 'completed\n' > "${STATUS}"
