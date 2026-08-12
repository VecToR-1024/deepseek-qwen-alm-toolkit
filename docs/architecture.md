# Architecture

## Collection plane

The campaign supervisor validates a frozen JSON configuration, worker/rate
budgets, disk headroom, supported data-source revisions, and required scripts
before any API call. Each lane imports a deterministic ordered task list and
then invokes the same breadth collector.

The streaming collector has three durable stages:

1. API workers generate attempts while one append writer owns
   `raw_attempts.jsonl`.
2. A tail reader normalizes new raw rows, validates schema and reconstructs
   `response_text.encode("utf-8")` from actual token bytes.
3. A bounded queue feeds isolated verifier workers; one writer owns
   `verifier_attempts.jsonl`.

Every stage recovers its terminal IDs from the corresponding JSONL file. A
restart therefore schedules only missing work. State snapshots are replaced
atomically and are observational; append-only queues remain authoritative.

## Trace profiles

`actual_only` preserves the complete request/response and the generated token
bytes/logprobs required by ALM. `top20` additionally preserves per-position
candidate distributions for the strict sparse baseline. Changing response text
after collection invalidates both profiles' probability alignment; cleaned
answers belong only in a separately labelled SFT dataset.

## Verification plane

The parent process performs structural checks but never imports generated code.
A child process receives extracted source and benchmark tests, uses a temporary
working directory, applies portable timeouts and Linux resource limits where
available, and reports separate compile/import/test outcomes. Durable failure
categories permit later analysis without feeding any verifier feedback back to
the teacher.

## Training plane

`OfflineTeacherTraceProvider` exposes actual teacher token bytes and logprobs
from normalized JSONL. `ALMExampleBuilder` renders the student chat prompt,
teacher-forces the same completion through the Qwen tokenizer, and constructs
ragged chunk metadata.

Teacher and student token byte streams are aligned at common byte endpoints in
O(T+S). Logprobs within a chunk are summed, then the trainer applies the stable
binary forward-KL ALM objective alongside causal hard SFT:

```text
total_loss = hard_sft_loss + alpha_alm * alm_loss
```

The strict one-token/top-20 implementation remains optional and separate. The
current model loader uses BF16/FP32 plus PEFT LoRA; it does not use bitsandbytes
or 4-bit QLoRA.

## Artifact contract

- Immutable inputs: pinned task JSONL, import summaries, generation config,
  model/tokenizer revisions, frozen training manifest.
- Append-only runtime truth: raw, normalization-error, normalized, and verifier
  JSONL queues.
- Derived outputs: accepted/rejected datasets, audit reports, ALM diagnostics,
  compressed shards, adapters, benchmark reports.
- Never committed: credentials, runtime data, model weights, checkpoints, logs.
