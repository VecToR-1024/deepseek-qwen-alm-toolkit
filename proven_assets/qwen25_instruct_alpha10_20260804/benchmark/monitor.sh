#!/usr/bin/env bash
set -u

: "${AUTODL_ROOT:?set AUTODL_ROOT to the remote workspace root}"

# Adapted from the completed hard-combined checkpoint benchmark dashboard.
RUN_ROOT=${AUTODL_ROOT}/benchmarks/qwen25_7b_instruct_hard_combined_alpha10_compare_v1_20260804
PY=${AUTODL_ROOT}/envs/topk-distill/bin/python
INTERVAL=10
ONCE=0
CANDIDATES=(base_qwen25_instruct qwen25_instruct_lora_step_256)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --interval)
            [[ $# -ge 2 && "$2" =~ ^[1-9][0-9]*$ ]] || exit 2
            INTERVAL="$2"
            shift 2
            ;;
        --once)
            ONCE=1
            shift
            ;;
        --help|-h)
            echo "Usage: monitor.sh [--interval SECONDS] [--once]"
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

line_count() {
    if [[ -f "$1" ]]; then
        wc -l < "$1"
    else
        printf 0
    fi
}

stage_status() {
    local candidate="$1"
    local stage="$2"
    local root="${RUN_ROOT}/stages/${candidate}"
    if [[ -f "${root}/${stage}.completed" ]]; then
        printf completed
    elif [[ -f "${root}/${stage}.status" ]]; then
        tr -d '\r\n' < "${root}/${stage}.status"
    elif [[ -f "${root}/${stage}.started_at" ]]; then
        printf running
    else
        printf pending
    fi
}

suite_status() {
    local suite="$1"
    local root="${RUN_ROOT}/stages/suites/${suite}"
    if [[ -f "${root}/completed" ]]; then
        printf completed
    elif [[ -f "${root}/exit_code" ]]; then
        printf 'stopped(exit=%s)' "$(<"${root}/exit_code")"
    elif [[ -f "${root}/started_at" ]]; then
        printf running
    else
        printf pending
    fi
}

render() {
    if [[ -t 1 && "${ONCE}" -eq 0 ]]; then clear; fi
    echo "Qwen2.5-7B-Instruct base/checkpoint benchmark"
    echo "Time: $(date --iso-8601=seconds)"
    if flock -n "${RUN_ROOT}/launcher.lock" true 2>/dev/null; then
        if [[ -f "${RUN_ROOT}/launcher.exit_code" ]]; then
            echo "Launcher: not running (latest exit $(<"${RUN_ROOT}/launcher.exit_code"))"
        else
            echo "Launcher: not running"
        fi
    else
        echo "Launcher: running"
    fi
    printf 'Suites: HumanEval+=%s  LiveCodeBench=%s\n' \
        "$(suite_status humaneval)" "$(suite_status livecodebench)"

    echo
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu \
        --format=csv,noheader 2>/dev/null || true

    echo
    echo "[HumanEval+]"
    for candidate in "${CANDIDATES[@]}"; do
        root="${RUN_ROOT}/humaneval/candidates/${candidate}/evalplus"
        raw="${root}/results/humaneval/model_hf_temp_0.0.raw.jsonl"
        summary="${root}/logs/humaneval_score_summary.json"
        printf '%-38s %3s/164 generated  codegen=%-12s score=%-12s' \
            "${candidate}" "$(line_count "${raw}")" \
            "$(stage_status "${candidate}" humaneval_codegen)" \
            "$(stage_status "${candidate}" humaneval_score)"
        if [[ -f "${summary}" ]]; then
            "${PY}" -c \
                'import json,sys;x=json.load(open(sys.argv[1]))["full"];print("  base={}/{} plus={}/{}".format(x["base_passes"],x["tasks"],x["plus_passes"],x["tasks"]),end="")' \
                "${summary}" 2>/dev/null || true
        fi
        echo
    done

    echo
    echo "[LiveCodeBench]"
    for candidate in "${CANDIDATES[@]}"; do
        root="${RUN_ROOT}/livecodebench/candidates/${candidate}"
        raw="${root}/strict/results/generations.jsonl"
        strict="${root}/strict/results/evaluation_summary_full.json"
        compatible="${root}/compatible/results/evaluation_summary_full.json"
        printf '%-38s %3s/339 generated  gen=%-12s strict=%-12s compat=%-12s' \
            "${candidate}" "$(line_count "${raw}")" \
            "$(stage_status "${candidate}" livecodebench_generate)" \
            "$(stage_status "${candidate}" livecodebench_strict_score)" \
            "$(stage_status "${candidate}" livecodebench_compatible_score)"
        if [[ -f "${strict}" ]]; then
            "${PY}" -c \
                'import json,sys;x=json.load(open(sys.argv[1]));print("  strict={}/{}".format(x["passed"],x["tasks"]),end="")' \
                "${strict}" 2>/dev/null || true
        fi
        if [[ -f "${compatible}" ]]; then
            "${PY}" -c \
                'import json,sys;x=json.load(open(sys.argv[1]));print(" compatible={}/{}".format(x["passed"],x["tasks"]),end="")' \
                "${compatible}" 2>/dev/null || true
        fi
        echo
    done

    echo
    printf 'Fatal patterns (all attempts): '
    grep -aERh \
        'Traceback|CUDA out of memory|No space left on device|AF_UNIX path too long|Permission denied' \
        "${RUN_ROOT}/stages" 2>/dev/null | wc -l
    df -h ${AUTODL_ROOT} | sed -n '2p'
    echo "Ctrl+C closes only this monitor. Use stage logs for verbose diagnostics."
}

while true; do
    render
    [[ "${ONCE}" -eq 1 ]] && break
    sleep "${INTERVAL}"
done
