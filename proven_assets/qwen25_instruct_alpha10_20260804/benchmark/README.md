# Qwen2.5-Instruct base/checkpoint benchmark

This durable launcher waits for the fresh training run to pass all gates, then
discovers every complete `checkpoint-*` directory. It compares the pinned base
model and all discovered LoRA checkpoints on the same frozen HumanEval+ and
LiveCodeBench protocols used in the previous phase.

LiveCodeBench is reported twice: the unchanged strict extraction score and the
deterministic `interface_wrapper` compatibility score. The same raw generations
feed both scores, so their difference measures formatting loss rather than a
second generation sample.

GPU generations are sequential. CPU scoring is isolated with the existing
non-root/seccomp harness. The launcher is append-only and stage-resumable and
does not start more training or select checkpoints post hoc.

The original supervisor design attempted to call `/sbin/poweroff` 30 seconds
after a terminal result. This cloud container is not booted with systemd, so
that command cannot stop the host. The supervisor is not armed for the resumed
run; stop the instance through the cloud control plane after completion. The
old `shutdown.reason=benchmark_pipeline_failed` marker is from the failed
2026-08-05 01:39 attempt and is not current run state.

## Current run

Training completed successfully at 08:34:35. The frozen comparison contains
only `base_qwen25_instruct` and `qwen25_instruct_lora_step_256`; the latter has
adapter SHA-256
`4951c85d10a518238406e1a7c6a60ea9157a590e1b809d28620ef3bc3d6ec6c4`.

The 08:35 benchmark attempt stopped before creating a scoring stage because
two deployed Python files had CRLF bytes and failed the pinned SHA-256 gate.
The drifted copies and attempt markers are archived under
`recovery/preflight_crlf_20260805T0835/`. After restoring the committed LF
files and verifying all ten preflight hashes, the launcher resumed at 09:18
without retraining.

The resumed monolithic launcher completed base HumanEval generation and
isolated scoring (`78/164` HumanEval, `72/164` HumanEval+) plus all 339 base
LiveCodeBench generations. It then exited `126` at 10:12 before strict scoring:
the attested `run_lcb_base_score.sh` bytes were correct, but the copied file was
not executable. The old completed LiveCodeBench launcher checked `test -x`;
the monolithic replacement checked only SHA-256.

The monolithic scripts are preserved under
`recovery/monolithic_exit126_20260805T101241/`. At 10:46 the run resumed with
the old proven structure:

```text
launch_after_training.sh
  -> launch_humaneval.sh
  -> launch_livecodebench.sh
```

The two suite launchers retain the old completed markers, latest-log pointers,
attempt history, pinned harnesses, and separate locks. Both score wrappers are
now checked for executable mode before a suite starts. Existing base HumanEval
and LCB generation stages are skipped; checkpoint-256 HumanEval generation is
currently running. No final comparison score is recorded yet.

Remote result root:

```text
${AUTODL_ROOT}/benchmarks/qwen25_7b_instruct_hard_combined_alpha10_compare_v1_20260804
```

Read-only monitor:

```powershell
.\monitor.ps1 -SshHost ssh.example.invalid -Port 2222 -User trainer -RemoteRoot /srv/alm
```

The monitor now reuses the completed checkpoint benchmark dashboard layout. It
shows suite state, per-candidate generation counts, stage status, GPU state,
available HumanEval+/strict/compatible scores, disk use, and fatal count. It
does not tail raw logs by default. See
[`ADR-005`](../../../docs/decisions/005-prefer-proven-launchers-and-visual-monitors.md).
