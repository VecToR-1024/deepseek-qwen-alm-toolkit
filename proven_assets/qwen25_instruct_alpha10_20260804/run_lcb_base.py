from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType


def required_autodl_root() -> Path:
    value = os.environ.get("AUTODL_ROOT", "").strip()
    if not value:
        raise RuntimeError("AUTODL_ROOT is required")
    return Path(value).expanduser()


AUTODL_ROOT = required_autodl_root()
BASELINE_ROOT = AUTODL_ROOT / "benchmarks/qwen25coder7b_livecodebench_v1"
LCB_SOURCE_ROOT = AUTODL_ROOT / "benchmarks/livecodebench-src"
PINNED_RUNNERS = {
    "generate": (
        BASELINE_ROOT / "generate.py",
        "dbb73d109c2e426f7ff9e6b745ed3873a6bf80eb2bd008a0113541b10c97aceb",
    ),
    "evaluate": (
        BASELINE_ROOT / "evaluate.py",
        "091360d84cc9d596d078d040482ebac317f25ac74516407312d795601d5a4542",
    ),
}
FROZEN_INPUTS = {
    "selected_dataset.jsonl": (
        "d7b9d4fb14931533c9b0f0be0577c27a912d4512e65b072899364a450ab5b751"
    ),
    "selection_manifest.json": (
        "15eb43fcceb5b01b23b8989185a68f81f8c4a4c1d56078b8e87859cb6c785605"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model_snapshot(path: Path, revision: str) -> None:
    resolved = path.resolve()
    if resolved.name != revision:
        raise ValueError(
            f"revision mismatch: snapshot={resolved.name!r} expected={revision!r}"
        )
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        if not (resolved / name).is_file():
            raise ValueError(f"missing model snapshot file: {name}")
    if not any(resolved.glob("*.safetensors")):
        raise ValueError("model snapshot has no safetensors weights")


def load_pinned_module(name: str) -> ModuleType:
    path, expected_hash = PINNED_RUNNERS[name]
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise RuntimeError(f"{name} runner SHA-256 mismatch: {actual_hash}")
    spec = importlib.util.spec_from_file_location(f"pinned_lcb_{name}_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pinned runner {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_frozen_inputs(run_root: Path) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    for filename, expected_hash in FROZEN_INPUTS.items():
        source = BASELINE_ROOT / filename
        if sha256_file(source) != expected_hash:
            raise RuntimeError(f"frozen input SHA-256 mismatch: {source}")
        target = run_root / filename
        if target.is_symlink():
            if target.resolve() != source.resolve():
                raise RuntimeError(f"{target} points to the wrong frozen input")
        elif target.exists():
            if sha256_file(target) != expected_hash:
                raise RuntimeError(f"existing frozen input differs: {target}")
        else:
            target.symlink_to(source)


def write_attestation(
    *,
    mode: str,
    candidate: str,
    base_model: Path,
    model_id: str,
    model_revision: str,
    run_root: Path,
) -> None:
    if mode == "generate":
        summary_path = run_root / "results" / "generation_summary.json"
        output_paths = [
            run_root / "results" / "generations.jsonl",
            run_root / "results" / "codegeneration_1_0.0.json",
            summary_path,
        ]
    else:
        suffix = "smoke" if os.environ.get("LCB_LIMIT", "").strip() else "full"
        summary_path = run_root / "results" / f"evaluation_summary_{suffix}.json"
        output_paths = [
            run_root / "results" / f"codegeneration_1_0.0_{suffix}_eval.json",
            run_root / "results" / f"codegeneration_1_0.0_{suffix}_eval_all.json",
            summary_path,
        ]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    attestation = {
        "schema_version": f"offline_alm.livecodebench.base_{mode}.v1",
        "candidate": candidate,
        "base_model": str(base_model),
        "model_id": model_id,
        "model_revision": model_revision,
        "runner_path": str(PINNED_RUNNERS[mode][0]),
        "runner_sha256": PINNED_RUNNERS[mode][1],
        "selected_dataset_sha256": FROZEN_INPUTS["selected_dataset.jsonl"],
        "summary": summary,
        "outputs": {
            str(path.relative_to(run_root)): sha256_file(path) for path in output_paths
        },
    }
    logs = run_root / "logs"
    logs.mkdir(exist_ok=True)
    (logs / f"{mode}_attestation.json").write_text(
        json.dumps(attestation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(attestation, ensure_ascii=False), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("generate", "evaluate"))
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "generate":
        args.base_model = args.base_model.resolve()
        validate_model_snapshot(args.base_model, args.model_revision)
    ensure_frozen_inputs(args.run_root)
    old_cwd = Path.cwd()
    try:
        os.chdir(LCB_SOURCE_ROOT)
        module = load_pinned_module(args.mode)
        module.RUN_ROOT = args.run_root
        if args.mode == "generate":
            module.MODEL_PATH = args.base_model
            module.MODEL_ID = args.model_id
            module.MODEL_REVISION = args.model_revision
        module.main()
    finally:
        os.chdir(old_cwd)
    write_attestation(
        mode=args.mode,
        candidate=args.candidate,
        base_model=args.base_model,
        model_id=args.model_id,
        model_revision=args.model_revision,
        run_root=args.run_root,
    )


if __name__ == "__main__":
    main()
