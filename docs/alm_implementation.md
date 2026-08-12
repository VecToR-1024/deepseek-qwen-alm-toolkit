# Offline DeepSeek-to-Qwen ALM

The primary cross-tokenizer objective is Approximate Likelihood Matching
(ALM). The implementation was adapted from the official
[`bminixhofer/tokenkit`](https://github.com/bminixhofer/tokenkit) repository at
commit `bcacdef43fb01397ff562ed397bddf9c20273194` and its
[`pytorch_alm_from_scratch.ipynb`](https://github.com/bminixhofer/tokenkit/blob/main/docs/pytorch_alm_from_scratch.ipynb)
guide. The corresponding implementation details were checked against
[`tokenkit/align.py`](https://github.com/bminixhofer/tokenkit/blob/main/tokenkit/align.py)
and
[`tokenkit/training/losses.py`](https://github.com/bminixhofer/tokenkit/blob/main/tokenkit/training/losses.py).
This project does not add tokenkit's JAX runtime as a dependency.

Only three ideas are reused:

1. Close an alignment chunk whenever teacher and student cumulative UTF-8 byte
   endpoints are equal.
2. Sum actual-token logprobs inside each chunk to obtain a chunk
   log-likelihood.
3. Compare the temperature-scaled chunk likelihoods as Bernoulli
   probabilities with a numerically stable binary divergence.

The two-pointer alignment advances at least one token pointer on every
iteration, so it is `O(T + S)` in teacher and student token counts, plus the
unavoidable linear pass over response bytes. It uses DeepSeek's returned byte
arrays and tokenizer offsets from the same Qwen encoding. Decoded-string
equality and `encode(decode(tokens))` are not alignment primitives.

## Objective

For aligned chunk `c`, the implementation computes:

```text
log q_c = sum of DeepSeek actual-token logprobs in c
log p_c = sum of Qwen actual-token logprobs in c
q'_c = exp(log q_c / temperature - epsilon)
p'_c = exp(log p_c / temperature - epsilon)
ALM_c = KL(Bernoulli(q'_c) || Bernoulli(p'_c))
total_loss = hard_sft_loss + alpha_alm * mean_c(ALM_c)
```

Qwen logprobs use teacher forcing on the complete assistant response. Logits at
position `t-1` score the actual token ID at position `t`; the final unused
logit is never gathered. `log(1-p)` uses the standard split between `log1p`
and `expm1` for stability near zero.

The notebook presents binary cross-entropy. Exact binary forward KL is used
here because it subtracts the fixed teacher Bernoulli entropy and therefore
has the same gradient with respect to the student while reporting a true
divergence with zero at equality.

## Offline teacher contract

`OfflineTeacherTraceProvider` reads a normalized DeepSeek record and returns:

- the complete `response_text`;
- every actually generated token's authoritative `bytes` array;
- every actually generated token's `logprob`.

The concatenated byte arrays must exactly equal `response_text.encode("utf-8")`.
No teacher model or teacher tokenizer is loaded. The top-20 alternatives stay
in the normalized record only for the optional strict baseline.

The Qwen chat template is rendered twice: once for the assistant generation
prefix and once for the complete training message. A student token crossing
the prompt/completion boundary is valid for hard SFT but is excluded from ALM,
because its full-token probability does not describe only the teacher's
completion bytes.

## Deliberately out of scope

- tokenkit outcome-chunk debiasing, `append_space`, and `merge_by_space_prob`;
- hidden-state or embedding distillation;
- tokenkit's JAX training stack;
- GOLD/ULD loss;
- top-20 counterfactual candidates in the primary ALM objective.

`PluggableDistillationTrainer`, the strict 1-to-1 candidate mapping, and the
top-20 plus tail-bucket math remain available as an experimental baseline.
