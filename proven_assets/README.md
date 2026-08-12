# Proven launcher snapshots

This directory preserves the control flow of selected launchers from completed
runs. Machine-specific roots and SSH endpoints have been removed; pinned
hashes, model revisions, and relative run layouts remain intact.

Set `AUTODL_ROOT` to the remote workspace root before running a shell or Python
launcher. Values containing `${AUTODL_ROOT}` in JSON are templates and must be
materialized by the caller before the manifest is consumed.

To reuse a launcher:

1. copy it into a new versioned run directory;
2. set `AUTODL_ROOT` and review every expected SHA-256 deliberately;
3. retain preflight, resume, lock, non-root scoring, and completion-marker
   behavior;
4. run shell syntax and one-task smoke checks;
5. keep deployment host, port, and user outside the repository.

`qwen25_instruct_alpha10_20260804/` contains the BF16 LoRA training and
base/checkpoint comparison assets. `evalplus_lcb_harness_20260729/` contains the
EvalPlus and LiveCodeBench generation/scoring harness used by later launchers.
