#!/usr/bin/env python3
"""Run a resumable, multi-source hard-task collection campaign locally."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from deepseek_distill.durable_io import replace_file
from deepseek_distill.hard_tasks import (
    HARD_DIFFICULTY_PROFILE,
    HARD_PROFILE_SOURCES,
)


CAMPAIGN_SCHEMA_VERSION = "coding.collection.hard_overnight.v1"
_WINDOWS_CREATE_NEW_PROCESS_GROUP = 0x00000200
_WINDOWS_ES_CONTINUOUS = 0x80000000
_WINDOWS_ES_SYSTEM_REQUIRED = 0x00000001


@dataclass(frozen=True, slots=True)
class LaneConfig:
    name: str
    source: str
    limit: int
    seed: int
    exclude_tasks: tuple[Path, ...]
    api_workers: int
    verifier_workers: int
    requests_per_minute: float


@dataclass(frozen=True, slots=True)
class CampaignConfig:
    path: Path
    repo_root: Path
    campaign_id: str
    run_root: Path
    cache_dir: Path
    minimum_free_gib: float
    stop_free_gib: float
    difficulty_profile: str | None
    generation: dict[str, Any]
    sampling: dict[str, Any]
    verification: dict[str, Any]
    budgets: dict[str, Any]
    lanes: tuple[LaneConfig, ...]
    hf_endpoint: str | None
    raw: dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def load_campaign_config(path: Path, *, repo_root: Path) -> CampaignConfig:
    path = Path(path).resolve()
    repo_root = Path(repo_root).resolve()
    raw = _read_json_object(path)
    if raw.get("schema_version") != CAMPAIGN_SCHEMA_VERSION:
        raise ValueError("campaign schema_version is incompatible")
    campaign_id = _required_text(raw.get("campaign_id"), "campaign_id")
    raw_difficulty_profile = raw.get("difficulty_profile")
    difficulty_profile = (
        None
        if raw_difficulty_profile is None
        else _required_text(raw_difficulty_profile, "difficulty_profile")
    )
    if difficulty_profile not in {None, HARD_DIFFICULTY_PROFILE}:
        raise ValueError(
            "difficulty_profile must be null or "
            f"{HARD_DIFFICULTY_PROFILE!r}"
        )
    run_root = _resolve_from_repo(raw.get("run_root"), repo_root, "run_root")
    cache_dir = _resolve_from_repo(raw.get("cache_dir"), repo_root, "cache_dir")
    minimum_free_gib = _positive_number(
        raw.get("minimum_free_gib"),
        "minimum_free_gib",
    )
    stop_free_gib = _positive_number(raw.get("stop_free_gib"), "stop_free_gib")
    if stop_free_gib >= minimum_free_gib:
        raise ValueError("stop_free_gib must be smaller than minimum_free_gib")
    generation = _required_mapping(raw.get("generation"), "generation")
    trace_profile = generation.get("trace_profile", "top20")
    if trace_profile not in {"top20", "actual_only"}:
        raise ValueError(
            "generation.trace_profile must be 'top20' or 'actual_only'"
        )
    generation["trace_profile"] = trace_profile
    sampling = _required_mapping(raw.get("sampling"), "sampling")
    max_attempts_per_task = _positive_int(
        sampling.get("max_attempts_per_task"),
        "sampling.max_attempts_per_task",
    )
    if max_attempts_per_task not in {1, 2, 3}:
        raise ValueError("sampling.max_attempts_per_task must be 1, 2, or 3")
    verification = _required_mapping(raw.get("verification"), "verification")
    budgets = _required_mapping(raw.get("budgets"), "budgets")
    lane_values = raw.get("lanes")
    if not isinstance(lane_values, list) or not lane_values:
        raise ValueError("lanes must be a non-empty list")
    lanes: list[LaneConfig] = []
    seen_names: set[str] = set()
    seen_sources: set[str] = set()
    for index, value in enumerate(lane_values):
        item = _required_mapping(value, f"lanes[{index}]")
        name = _required_text(item.get("name"), f"lanes[{index}].name")
        source = _required_text(item.get("source"), f"lanes[{index}].source")
        if name in seen_names:
            raise ValueError(f"duplicate lane name {name!r}")
        if source in seen_sources:
            raise ValueError(f"duplicate lane source {source!r}")
        if source not in HARD_PROFILE_SOURCES:
            raise ValueError(f"hard campaign does not support source {source!r}")
        seen_names.add(name)
        seen_sources.add(source)
        lanes.append(
            LaneConfig(
                name=name,
                source=source,
                limit=_positive_int(item.get("limit"), f"lanes[{index}].limit"),
                seed=_integer(item.get("seed"), f"lanes[{index}].seed"),
                exclude_tasks=_optional_paths_from_repo(
                    item.get("exclude_tasks"),
                    repo_root,
                    f"lanes[{index}].exclude_tasks",
                ),
                api_workers=_positive_int(
                    item.get("api_workers"),
                    f"lanes[{index}].api_workers",
                ),
                verifier_workers=_positive_int(
                    item.get("verifier_workers"),
                    f"lanes[{index}].verifier_workers",
                ),
                requests_per_minute=_positive_number(
                    item.get("requests_per_minute"),
                    f"lanes[{index}].requests_per_minute",
                ),
            )
        )
    _validate_budget(
        sum(lane.api_workers for lane in lanes),
        budgets,
        "max_total_api_workers",
        "API worker budget",
    )
    _validate_budget(
        sum(lane.verifier_workers for lane in lanes),
        budgets,
        "max_total_verifier_workers",
        "verifier worker budget",
    )
    _validate_budget(
        sum(lane.requests_per_minute for lane in lanes),
        budgets,
        "max_total_requests_per_minute",
        "request-rate budget",
    )
    hf_endpoint = raw.get("hf_endpoint")
    if hf_endpoint is not None:
        hf_endpoint = _required_text(hf_endpoint, "hf_endpoint")
    return CampaignConfig(
        path=path,
        repo_root=repo_root,
        campaign_id=campaign_id,
        run_root=run_root,
        cache_dir=cache_dir,
        minimum_free_gib=minimum_free_gib,
        stop_free_gib=stop_free_gib,
        difficulty_profile=difficulty_profile,
        generation=generation,
        sampling=sampling,
        verification=verification,
        budgets=budgets,
        lanes=tuple(lanes),
        hf_endpoint=hf_endpoint,
        raw=raw,
    )


def lane_paths(config: CampaignConfig, lane: LaneConfig) -> dict[str, Path]:
    return {
        "tasks": config.run_root / "import" / f"{lane.name}_tasks_{lane.limit}.jsonl",
        "import_summary": (
            config.run_root / "import" / f"{lane.name}_import_{lane.limit}.json"
        ),
        "run_dir": config.run_root / "runs" / lane.name,
        "log": config.run_root / "logs" / f"{lane.name}.log",
    }


def build_import_command(
    config: CampaignConfig,
    lane: LaneConfig,
    *,
    python: str,
) -> list[str]:
    paths = lane_paths(config, lane)
    command = [
        python,
        str(config.repo_root / "scripts/import_multisource.py"),
        "--source",
        lane.source,
        "--limit",
        str(lane.limit),
        "--selection",
        "random",
        "--seed",
        str(lane.seed),
        "--cache-dir",
        str(config.cache_dir),
        "--output",
        str(paths["tasks"]),
        "--summary-output",
        str(paths["import_summary"]),
    ]
    if config.difficulty_profile is not None:
        command.extend(("--difficulty-profile", config.difficulty_profile))
    for exclusion_path in lane.exclude_tasks:
        command.extend(("--exclude-tasks", str(exclusion_path)))
    return command


def build_collect_command(
    config: CampaignConfig,
    lane: LaneConfig,
    *,
    python: str,
) -> list[str]:
    paths = lane_paths(config, lane)
    generation = config.generation
    verification = config.verification
    command = [
        python,
        str(config.repo_root / "scripts/collect_multisource_breadth.py"),
        "--source",
        lane.source,
        "--tasks",
        str(paths["tasks"]),
        "--import-summary",
        str(paths["import_summary"]),
        "--run-dir",
        str(paths["run_dir"]),
        "--model",
        str(generation["model"]),
        "--workers",
        str(lane.api_workers),
        "--requests-per-minute",
        str(lane.requests_per_minute),
        "--timeout",
        str(generation["timeout"]),
        "--max-retries",
        str(generation["max_retries"]),
        "--temperature",
        str(generation["temperature"]),
        "--top-p",
        str(generation["top_p"]),
        "--trace-profile",
        str(generation["trace_profile"]),
    ]
    if generation["trace_profile"] == "top20":
        command.extend(("--top-logprobs", str(generation["top_logprobs"])))
    command.extend(
        [
            "--max-tokens",
            str(generation["max_tokens"]),
            "--max-attempts-per-task",
            str(config.sampling["max_attempts_per_task"]),
            "--phase-timeout",
            str(verification["phase_timeout"]),
            "--max-output-characters",
            str(verification["max_output_characters"]),
            "--verifier-workers",
            str(lane.verifier_workers),
            "--streaming-pipeline",
        ]
    )
    return command


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_campaign_config(args.config, repo_root=args.repo_root)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "campaign_id": config.campaign_id,
                    "commands": {
                        lane.name: {
                            "import": build_import_command(config, lane, python=args.python),
                            "collect": build_collect_command(config, lane, python=args.python),
                        }
                        for lane in config.lanes
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    _preflight(config, python=args.python)
    config.run_root.mkdir(parents=True, exist_ok=True)
    for name in ("import", "logs", "runs"):
        (config.run_root / name).mkdir(exist_ok=True)
    _freeze_config(config)
    _write_json_atomic(
        config.run_root / "supervisor.json",
        {
            "schema_version": CAMPAIGN_SCHEMA_VERSION,
            "campaign_id": config.campaign_id,
            "pid": os.getpid(),
            "started_at": _now(),
            "python": str(Path(args.python).resolve()),
            "repo_root": str(config.repo_root),
            "training_started": False,
        },
    )

    stop_path = config.run_root / "STOP"
    stop_path.unlink(missing_ok=True)
    _keep_windows_awake(True)
    lane_states: dict[str, dict[str, Any]] = {
        lane.name: {"status": "pending", "exit_code": None, "pid": None}
        for lane in config.lanes
    }
    state_lock = threading.Lock()
    threads: list[threading.Thread] = []
    try:
        for lane in config.lanes:
            thread = threading.Thread(
                target=_run_lane,
                args=(config, lane, args.python, stop_path, lane_states, state_lock),
                name=f"hard-collector-{lane.name}",
                daemon=False,
            )
            thread.start()
            threads.append(thread)

        while any(thread.is_alive() for thread in threads):
            free_gib = _free_gib(config)
            if free_gib < config.stop_free_gib and not stop_path.exists():
                stop_path.write_text(
                    json.dumps(
                        {
                            "reason": "low_disk_space",
                            "free_gib": free_gib,
                            "threshold_gib": config.stop_free_gib,
                            "at": _now(),
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            _publish_supervisor_state(config, lane_states, state_lock, stopped=stop_path.exists())
            time.sleep(2.0)
        for thread in threads:
            thread.join()
        stopped = stop_path.exists()
        _publish_supervisor_state(config, lane_states, state_lock, stopped=stopped)
        failed = [
            name
            for name, state in lane_states.items()
            if state.get("exit_code") not in {0, None}
        ]
        status = "stopped" if stopped else ("failed" if failed else "complete")
        result = {
            "schema_version": CAMPAIGN_SCHEMA_VERSION,
            "campaign_id": config.campaign_id,
            "status": status,
            "finished_at": _now(),
            "lanes": lane_states,
            "training_started": False,
        }
        _write_json_atomic(config.run_root / f"campaign.{status}.json", result)
        return 0 if status == "complete" else (130 if status == "stopped" else 2)
    finally:
        _keep_windows_awake(False)


def _run_lane(
    config: CampaignConfig,
    lane: LaneConfig,
    python: str,
    stop_path: Path,
    states: dict[str, dict[str, Any]],
    lock: threading.Lock,
) -> None:
    paths = lane_paths(config, lane)
    paths["log"].parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(config.repo_root / "src")
    with paths["log"].open("a", encoding="utf-8", newline="\n") as log:
        try:
            for phase, command in (
                ("importing", build_import_command(config, lane, python=python)),
                ("collecting", build_collect_command(config, lane, python=python)),
            ):
                if stop_path.exists():
                    _set_lane_state(states, lock, lane.name, status="stopped", exit_code=130)
                    return
                child_environment = dict(environment)
                if phase == "importing" and config.hf_endpoint:
                    child_environment["HF_ENDPOINT"] = config.hf_endpoint
                _log_event(log, lane=lane.name, event=f"{phase}_start", command=command)
                process = subprocess.Popen(
                    command,
                    cwd=config.repo_root,
                    env=child_environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    creationflags=(
                        _WINDOWS_CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
                    ),
                )
                _set_lane_state(
                    states,
                    lock,
                    lane.name,
                    status=phase,
                    pid=process.pid,
                    exit_code=None,
                )
                exit_code = _wait_for_process(process, stop_path=stop_path)
                _log_event(
                    log,
                    lane=lane.name,
                    event=f"{phase}_exit",
                    exit_code=exit_code,
                )
                if exit_code != 0:
                    status = "stopped" if stop_path.exists() else "failed"
                    _set_lane_state(
                        states,
                        lock,
                        lane.name,
                        status=status,
                        pid=None,
                        exit_code=exit_code,
                    )
                    return
            _set_lane_state(
                states,
                lock,
                lane.name,
                status="complete",
                pid=None,
                exit_code=0,
            )
        except BaseException as error:
            _log_event(
                log,
                lane=lane.name,
                event="lane_exception",
                error={"type": type(error).__name__, "message": str(error)},
            )
            _set_lane_state(
                states,
                lock,
                lane.name,
                status="failed",
                pid=None,
                exit_code=1,
                error={"type": type(error).__name__, "message": str(error)},
            )


def _wait_for_process(process: subprocess.Popen[str], *, stop_path: Path) -> int:
    while True:
        exit_code = process.poll()
        if exit_code is not None:
            return exit_code
        if stop_path.exists():
            _terminate_process_tree(process)
            return process.wait()
        time.sleep(1.0)


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()


def _preflight(config: CampaignConfig, *, python: str) -> None:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY is missing; refusing to call the API")
    if not Path(python).is_file():
        raise FileNotFoundError(f"Python interpreter does not exist: {python}")
    for script in (
        config.repo_root / "scripts/import_multisource.py",
        config.repo_root / "scripts/collect_multisource_breadth.py",
    ):
        if not script.is_file():
            raise FileNotFoundError(f"required script is missing: {script}")
    for lane in config.lanes:
        for exclusion_path in lane.exclude_tasks:
            if not exclusion_path.is_file():
                raise FileNotFoundError(
                    f"exclusion task artifact is missing: {exclusion_path}"
                )
    free_gib = _free_gib(config)
    if free_gib < config.minimum_free_gib:
        raise RuntimeError(
            f"only {free_gib:.1f} GiB free; campaign requires "
            f"{config.minimum_free_gib:.1f} GiB"
        )


def _freeze_config(config: CampaignConfig) -> None:
    snapshot = config.run_root / "campaign_config.snapshot.json"
    serialized = json.dumps(config.raw, ensure_ascii=False, indent=2) + "\n"
    if snapshot.exists():
        if snapshot.read_text(encoding="utf-8") != serialized:
            raise FileExistsError("campaign config snapshot differs from existing run")
        return
    snapshot.write_text(serialized, encoding="utf-8", newline="\n")


def _publish_supervisor_state(
    config: CampaignConfig,
    states: dict[str, dict[str, Any]],
    lock: threading.Lock,
    *,
    stopped: bool,
) -> None:
    with lock:
        lanes = json.loads(json.dumps(states))
    pipeline_states: dict[str, Any] = {}
    for lane in config.lanes:
        state_path = lane_paths(config, lane)["run_dir"] / "pipeline_state.json"
        if state_path.exists():
            try:
                pipeline_states[lane.name] = _read_json_object(state_path)
            except (OSError, ValueError):
                pipeline_states[lane.name] = {"phase": "state_temporarily_unreadable"}
    _write_json_atomic(
        config.run_root / "supervisor_state.json",
        {
            "schema_version": CAMPAIGN_SCHEMA_VERSION,
            "campaign_id": config.campaign_id,
            "updated_at": _now(),
            "stop_requested": stopped,
            "lanes": lanes,
            "pipelines": pipeline_states,
            "disk_free_gib": round(_free_gib(config), 2),
        },
    )


def _set_lane_state(
    states: dict[str, dict[str, Any]],
    lock: threading.Lock,
    lane: str,
    **updates: Any,
) -> None:
    with lock:
        states[lane].update(updates)
        states[lane]["updated_at"] = _now()


def _keep_windows_awake(enabled: bool) -> None:
    if os.name != "nt":
        return
    flags = (
        _WINDOWS_ES_CONTINUOUS | _WINDOWS_ES_SYSTEM_REQUIRED
        if enabled
        else _WINDOWS_ES_CONTINUOUS
    )
    ctypes.windll.kernel32.SetThreadExecutionState(flags)


def _log_event(handle: Any, **event: Any) -> None:
    event["at"] = _now()
    handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    handle.flush()


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    replace_file(temporary, path)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON: {error.msg}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _required_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _resolve_from_repo(value: Any, repo_root: Path, label: str) -> Path:
    text = _required_text(value, label)
    path = Path(text)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _optional_paths_from_repo(
    value: Any,
    repo_root: Path,
    label: str,
) -> tuple[Path, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return tuple(
        _resolve_from_repo(item, repo_root, f"{label}[{index}]")
        for index, item in enumerate(value)
    )


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _positive_int(value: Any, label: str) -> int:
    result = _integer(value, label)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _positive_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{label} must be a positive number")
    return float(value)


def _validate_budget(
    actual: float,
    budgets: Mapping[str, Any],
    key: str,
    label: str,
) -> None:
    maximum = _positive_number(budgets.get(key), f"budgets.{key}")
    if actual > maximum:
        raise ValueError(f"{label} exceeded: {actual} > {maximum}")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _free_gib(config: CampaignConfig) -> float:
    return shutil.disk_usage(config.run_root.anchor).free / 2**30


if __name__ == "__main__":
    raise SystemExit(main())
