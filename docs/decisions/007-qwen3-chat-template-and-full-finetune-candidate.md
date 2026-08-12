# ADR-007: Make chat-template arguments explicit for Qwen3

## Status

Accepted

## Date

2026-08-12

## Context

Qwen3 can render different generation contexts depending on the thinking-mode
template argument. The offline teacher trace is aligned against the exact
student prompt/completion bytes, so using a different setting during audit or
training breaks the prompt/completion boundary even when the response text is
unchanged. A read-only data acceptance run proved the current records compatible
with `enable_thinking=false`.

The next candidate base is Qwen3-0.6B. A 48GB GPU is sufficient to explore a
BF16 full-parameter run, so requiring PEFT/LoRA is unnecessary for that branch.

## Decision

Allow explicit JSON chat-template kwargs in the ALM builder, training entrypoint,
and frozen-dataset audit. Defaults remain empty to preserve existing Qwen2.5
behavior. Pin the Qwen3 candidate revision and set `USE_LORA=0` in a separate
launcher/config rather than changing the proven Qwen2.5 LoRA assets.

## Consequences

- Qwen3 prompt and completion rendering use one auditable non-thinking contract.
- The full-finetune path imports PEFT only when LoRA is actually requested.
- Existing Qwen2.5 behavior and ALM math are unchanged.
- The Qwen3 launcher remains a candidate until a real GPU smoke passes; it does
  not belong in `proven_assets/` yet.
