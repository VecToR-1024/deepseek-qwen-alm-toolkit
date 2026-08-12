#!/usr/bin/env bash
set -euo pipefail

: "${AUTODL_ROOT:?set AUTODL_ROOT to the remote workspace root}"

RUN_ROOT=${AUTODL_ROOT}/experiments/qwen25_7b_instruct_hard_combined_alpha10_v1_20260804

date --iso-8601=seconds
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader 2>/dev/null || echo "GPU unavailable"
for stage in model_snapshot base_benchmark_smoke one_epoch_training; do
    root="${RUN_ROOT}/stages/${stage}"
    status=pending
    [[ -f "${root}/${stage}.status" ]] && status="$(<"${root}/${stage}.status")"
    printf '%s: %s\n' "${stage}" "${status}"
    latest=$(find "${root}" -maxdepth 1 -type f -name "${stage}.*.log" 2>/dev/null \
        | sort | tail -n 1 || true)
    if [[ -n "${latest}" ]]; then
        echo "--- ${latest} ---"
        tail -n 8 "${latest}"
    fi
done
for log in \
    "${RUN_ROOT}/logs/smoke/train.log" \
    "${RUN_ROOT}/logs/formal/train.log"; do
    if [[ -f "${log}" ]]; then
        echo "--- ${log} ---"
        tail -n 8 "${log}"
    fi
done
if [[ -f "${RUN_ROOT}/launcher.exit_code" ]]; then
    printf 'launcher_exit=%s\n' "$(<"${RUN_ROOT}/launcher.exit_code")"
else
    echo "launcher_exit=running_or_not_started"
fi
df -h ${AUTODL_ROOT} | tail -n 1
