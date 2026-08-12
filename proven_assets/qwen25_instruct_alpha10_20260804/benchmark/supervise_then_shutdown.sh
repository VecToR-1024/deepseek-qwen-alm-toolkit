#!/usr/bin/env bash
set -euo pipefail

: "${AUTODL_ROOT:?set AUTODL_ROOT to the remote workspace root}"

RUN_ROOT=${AUTODL_ROOT}/benchmarks/qwen25_7b_instruct_hard_combined_alpha10_compare_v1_20260804
POLL_SECONDS="${POLL_SECONDS:-30}"
shutdown_on_terminal_failure=true

mkdir -p "${RUN_ROOT}/supervisor"
exec 9>"${RUN_ROOT}/supervisor.lock"
if ! flock -n 9; then
    echo "another shutdown supervisor already holds the lock" >&2
    exit 75
fi

finish_supervisor() {
    status=$?
    printf '%s\n' "${status}" > "${RUN_ROOT}/supervisor/exit_code"
    date --iso-8601=seconds > "${RUN_ROOT}/supervisor/finished_at"
}
trap finish_supervisor EXIT

schedule_shutdown() {
    local terminal_status="$1"
    local reason="$2"
    printf '%s\n' "${terminal_status}" > "${RUN_ROOT}/pipeline.exit_code"
    printf '%s\n' "${reason}" > "${RUN_ROOT}/shutdown.reason"
    date --iso-8601=seconds > "${RUN_ROOT}/shutdown.requested_at"
    sync
    nohup bash -c 'sleep 30; /sbin/poweroff' \
        > "${RUN_ROOT}/shutdown.log" 2>&1 &
    printf '%s\n' "$!" > "${RUN_ROOT}/shutdown.pid"
    sync
}

# These checks are non-destructive. The supervisor is armed only when the
# already-running benchmark launcher owns its lock and poweroff is available.
test "${POLL_SECONDS}" -ge 1
test -x /sbin/poweroff
test -x "${RUN_ROOT}/launch_after_training.sh"
if flock -n "${RUN_ROOT}/launcher.lock" true 2>/dev/null; then
    echo "benchmark launcher lock is not held; refusing to arm shutdown" >&2
    exit 2
fi
if [[ -f "${RUN_ROOT}/shutdown.requested_at" ]]; then
    echo "shutdown was already requested; refusing to arm twice" >&2
    exit 76
fi
date --iso-8601=seconds > "${RUN_ROOT}/supervisor/started_at"
date --iso-8601=seconds > "${RUN_ROOT}/shutdown.armed_at"

while true; do
    date --iso-8601=seconds > "${RUN_ROOT}/supervisor/heartbeat"
    if flock -n "${RUN_ROOT}/launcher.lock" true 2>/dev/null; then
        if [[ ! -f "${RUN_ROOT}/launcher.exit_code" ]]; then
            schedule_shutdown 95 "benchmark_launcher_ended_without_status"
            exit 95
        fi
        status="$(<"${RUN_ROOT}/launcher.exit_code")"
        if [[ ! "${status}" =~ ^[0-9]+$ ]]; then
            schedule_shutdown 94 "benchmark_launcher_status_invalid"
            exit 94
        fi
        if [[ "${status}" == 0 ]] && [[ -f "${RUN_ROOT}/all.completed_at" ]]; then
            schedule_shutdown 0 "all_benchmarks_completed"
            exit 0
        fi
        if [[ "${status}" == 0 ]]; then
            status=96
        fi
        schedule_shutdown "${status}" "benchmark_pipeline_failed"
        exit "${status}"
    fi
    sleep "${POLL_SECONDS}"
done
