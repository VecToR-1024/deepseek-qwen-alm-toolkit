# Qwen2.5-7B-Instruct hard-combined ALM v1

This is a new experiment family. It replaces the Coder base with
`Qwen/Qwen2.5-7B-Instruct` while reusing the frozen 2,041 offline DeepSeek
teacher traces. Student tokenization, labels, and ALM chunks are rebuilt with
the new tokenizer and chat template; no old adapter or checkpoint is loaded.

The model revision is pinned to
`a09a35458c702b33eeacc393d103063234e8bc28`, obtained from the official
[Qwen model repository](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct).
The pinned LiveCodeBench source already registers this model with the
`CodeQwenInstruct` prompt style.

The automatic pipeline is deliberately gated:

1. download and attest the pinned model snapshot;
2. run three-task HumanEval+ and LiveCodeBench base smoke scores;
3. audit all 2,041 examples with the new tokenizer;
4. run one GPU optimizer-step smoke;
5. train one epoch from the new base with BF16 LoRA and `alpha_alm=10`.

It does not run the full benchmark suite, start a second epoch, reuse a Coder
checkpoint, or shut down the server automatically.

## Completed training

All training gates passed. The 2,041-example preflight found supervised EOS on
every record, zero ALM boundary drops, zero zero-chunk records, no fences, no
overlength records, and no exact frozen-benchmark problem overlap. The formal
run completed 256/256 optimizer steps at epoch 1 on 2026-08-05 08:34:35+08:00.
Trainer runtime was 1,310 seconds.

The frozen checkpoint is:

```text
${AUTODL_ROOT}/experiments/qwen25_7b_instruct_hard_combined_alpha10_v1_20260804/outputs/alpha10/checkpoint-256
```

Its adapter SHA-256 is
`4951c85d10a518238406e1a7c6a60ea9157a590e1b809d28620ef3bc3d6ec6c4`.
Training loss is not used for checkpoint promotion; selection waits for the
full benchmark comparison.

## Recovery notes

The first three-task HumanEval score failed because the full 164-task EvalPlus
scorer rejects partial samples. Commit `797c8a4` added an exact frozen-subset
wrapper around the pinned evaluator; commit `411f065` added explicit null
adapter metadata for base-model summaries. The repaired smoke score was 2/3
for both HumanEval and HumanEval+ and all LiveCodeBench smoke stages passed.

The first post-training benchmark launch then stopped at its checksum gate
because two copied Python files had CRLF line endings. Their original bytes and
the failed launcher evidence are preserved under the benchmark `recovery/`
directory. The attested LF files were restored and the benchmark resumed
without retraining.

That resumed monolithic launcher completed base HumanEval generation/scoring
and all 339 base LiveCodeBench generations, then exited `126` at 10:12 because
`run_lcb_base_score.sh` had lost its executable bit during upload. The script
bytes and SHA-256 were correct. The failed launcher was archived, and the run
was switched back to the previously successful master/HumanEval/LiveCodeBench
launcher layout. Both score wrappers now have an executable-bit preflight;
shell assets are deployed as `0755`. The reused launcher resumed at 10:46 and
skips all completed base stages.

The cloud container does not run systemd. `/sbin/poweroff` therefore cannot
stop the host; automatic shutdown is not available for this run.

Remote root:

```text
${AUTODL_ROOT}/experiments/qwen25_7b_instruct_hard_combined_alpha10_v1_20260804
```

Read-only status:

```powershell
.\monitor.ps1 -SshHost ssh.example.invalid -Port 2222 -User trainer -RemoteRoot /srv/alm
```

Full benchmark status:

```powershell
.\benchmark\monitor.ps1 -SshHost ssh.example.invalid -Port 2222 -User trainer -RemoteRoot /srv/alm
```

Future orchestration and monitoring changes follow
[`ADR-005`](../../docs/decisions/005-prefer-proven-launchers-and-visual-monitors.md):
reuse a proven launcher/monitor first, add only thin parameter adapters, and
keep raw log tails out of the default monitor view.
