#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${1:-${OUTPUT_ROOT:-outputs/comparison_v1}}"
date --iso-8601=seconds
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader 2>/dev/null || echo "GPU unavailable"
for arm in sft_only sft_alm; do
    root="${OUTPUT_ROOT}/${arm}"
    if [[ -f "${root}/completed" ]]; then
        status=completed
    elif [[ -f "${root}/exit_code" ]]; then
        status="exit:$(<"${root}/exit_code")"
    elif [[ -f "${root}/started_at" ]]; then
        status=running
    else
        status=pending
    fi
    printf '%s: %s\n' "${arm}" "${status}"
    if [[ -f "${root}/train.log" ]]; then
        tail -n 6 "${root}/train.log"
    fi
done
