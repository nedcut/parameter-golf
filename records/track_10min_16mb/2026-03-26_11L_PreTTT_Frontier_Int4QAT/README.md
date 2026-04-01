# 11L Pre-TTT Frontier + Int4 QAT Scaffold

This folder is the local catch-up path for starting from the March 22 competitive codebase, first retuning it toward the March 25 no-TTT frontier, then testing the new int4 late-onset Hadamard/trust-gradient QAT on top of that stronger recipe.

It is intentionally **not** a historical record submission. It is a working scaffold for:

- matching the March 25 no-TTT stack as closely as possible
- keeping the competitive work out of the beginner-oriented top-level `train_gpt.py`
- comparing three training-time quantization modes on the same frontier stack

## Base Stack

Implementation base:

- March 22 `11L EMA + GPTQ-lite + warmdown3500 + QAT@0.15`

Already-applied retunes beyond the plain March 22 base:

- `LeakyReLU(0.5)^2` MLP activation
- `BIGRAM_VOCAB_SIZE=1536` default
- `GPTQ-lite int6 + lzma` export path
- preserve EMA, tight SWA, XSA, partial RoPE, VE128, seq_len 2048, eval stride 64

Still not equivalent to the current March 25 frontier:

- no Parameter Banking + Parallel Muon yet
- still on XSA-last-4 rather than XSA-all
- still on `BIGRAM_VOCAB_SIZE=1536` rather than `3072x112`
- still on warmdown 3500 rather than 4000
- still on GPTQ-lite rather than the March 25 full-Hessian / self-generated-calibration path

March 23 remains a useful intermediate waypoint, but it is no longer the main target.

## QAT Modes

### 1. No QAT

```bash
QAT_BITS=0 \
QAT_ENABLED=0 \
LATE_QAT_THRESHOLD=0 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### 2. Legacy Late Int6 Fake Quant

```bash
QAT_BITS=0 \
QAT_ENABLED=0 \
LATE_QAT_THRESHOLD=0.15 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### 3. New Int4 Late-Onset Hadamard / Trust-Gradient QAT

```bash
QAT_BITS=4 \
QAT_ENABLED=0 \
QAT_ONSET_SCALE=0.15 \
QAT_BLOCK_SIZE=128 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Default Frontier Settings

- 11 layers, 512 width, 8 heads, 4 KV heads
- 3x MLP
- `LeakyReLU(0.5)^2`
- BigramHash 1536
- XSA on last 4 layers
- partial RoPE 16/64
- VE128 on layers 9 and 10
- EMA decay 0.997
- tight SWA every 50 when LR scale < 0.2
- warmdown 3500
- seq_len 2048
- train batch 786,432 tokens
- Muon/Adam weight decay 0.04
- sliding-window eval stride 64
- GPTQ-lite int6 export with lzma

## Acceptance Targets

- Base sanity target: around March 22 behavior (`~1.1228 val_bpb` single-seed, `~1.1233` three-seed mean)
- Intermediate waypoint after the early retunes: around the March 23 pre-TTT result (`~1.1218 val_bpb`)
- Main parity target after the March 25 ingredients are stacked: around `1.1147 val_bpb`
- First int4-QAT success bar:
  - mean pre-TTT improvement of at least `0.0005 BPB` over the matched no-QAT stack, or
  - materially better artifact budget that justifies a larger-model follow-up

## Notes

- `QAT_BITS=4` is the only supported value for the new Hadamard/trust-gradient path.
- `QAT_ONSET_SCALE` defaults to `0.15` on this scaffold because that is already validated in the public late-QAT record stack.
- For int4 runs, the wrapper maps `QAT_ONSET_SCALE` onto the inherited late-QAT scheduler so the runtime gate actually turns on when warmdown reaches the requested scale.
- The wrapper prints a `frontier_scaffold:` preflight line showing the resolved QAT mode before delegating to the inherited March 22 trainer.
- For 1-GPU smoke tests, use `torchrun --standalone --nproc_per_node=1 train_gpt.py`.
- For 200-step smoke tests, set `WARMDOWN_ITERS=$ITERATIONS` so "late-onset" behavior is actually late inside the short run.
- Set `FRONTIER_PRE_EMA_EXPORT_DIAGNOSTIC=1` to emit `pre_ema_*` metrics and compare raw endpoint export against the EMA endpoint.
- Use `./scripts/submit_frontier_matrix.sh` to launch the no-QAT, legacy-int6, and int4 smoke matrix across multiple seeds.
- Use `python3 scripts/summarize_frontier_logs.py "slurm/output/pg-frontier-smoke-*.out"` to summarize the resulting logs as a table.
- Run `python3 trust_gradient_check.py` for a lightweight regression check of the trust-gradient masking semantics.
- Best next parity deltas to port are: XSA-all, BigramHash `3072x112`, warmdown 4000, then Parameter Banking + Parallel Muon.
