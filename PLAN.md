# Parameter Golf Plan

Goal: bring this repo to pre-TTT frontier parity, then test int4 late-onset Hadamard/trust-gradient QAT on that stronger stack.

## Context

- **Challenge**: best LM in a 16MB artifact, trained in <=10 minutes on 8xH100s, scored on FineWeb val BPB
- **Deadline**: April 30, 2026
- **Current 10-minute SOTA**: 1.1194 val_bpb (`LeakyReLU^2 + Legal Score-First TTT + Parallel Muon`, 3-seed mean)
- **Current pre-TTT stretch target**: ~1.1218 val_bpb from the March 23 record stack before legal TTT
- **Nearest reproducible codebase checkpoint**: March 22 `11L EMA + GPTQ-lite + warmdown3500 + QAT@0.15` record stack

Late QAT is already part of the public frontier. Our new int4 Hadamard/trust-gradient QAT should be treated as a new ingredient to stack on top of the best pre-TTT recipe, not as the only missing idea.

## Gap Analysis

The top-level `train_gpt.py` remains a strong starter trainer, but it still lacks several frontier ingredients:

| Area | Starter trainer | Frontier stack | Where to implement |
|------|-----------------|----------------|--------------------|
| MLP activation | `relu^2` | `LeakyReLU(0.5)^2` | competitive `records/` path |
| MLP width | 2x | 3x | competitive `records/` path |
| Context | 1024 train/eval | 2048 train/eval | competitive `records/` path |
| Positional setup | full RoPE | partial RoPE 16/64 | competitive `records/` path |
| Attention tweak | none | XSA on last 4 layers | competitive `records/` path |
| Token identity features | none | BigramHash + VE128 | competitive `records/` path |
| Weight averaging | none | EMA + tight SWA | competitive `records/` path |
| Export path | int8 + zlib | GPTQ-lite int6 + lzma | reusable quant helpers ok, full path in `records/` |
| Weight decay | none | Muon/Adam weight decay 0.04 | competitive `records/` path |
| Eval mode | fixed-window | sliding-window stride 64 | competitive `records/` path |
| Optimizer systems | classic Muon | Parameter Banking + Parallel Muon | later competitive `records/` path |
| New QAT idea | int4 Hadamard/trust-gradient in starter script | not yet stacked on frontier | shared logic plus competitive `records/` path |

Decision: keep the beginner-oriented top-level trainer mostly stable. Competitive catch-up work should land in a new `records/track_10min_16mb/...` path, except for small reusable utilities that are genuinely shared.

## Current State

- [x] Leaderboard snapshot verified against upstream as of March 26, 2026
- [x] New int4 late-onset Hadamard/trust-gradient QAT implemented and hardened in top-level `train_gpt.py`
- [x] Internal docs updated to reflect the real frontier
- [x] New pre-TTT frontier scaffold created under `records/track_10min_16mb/2026-03-26_11L_PreTTT_Frontier_Int4QAT`
- [ ] 1-GPU CUDA smoke test on the new frontier scaffold
- [ ] Pre-TTT parity check against the March 23 stack
- [ ] Int4-QAT ablation on the parity stack
- [ ] Int4 export / larger-model follow-up if int4 QAT shows value
- [ ] Legal score-first TTT after pre-TTT parity is stable

## Roadmap

### Phase 1 - Pre-TTT Frontier Parity

Build and validate a reproducible pre-TTT frontier path in a new record-style folder based on the March 22 competitive script, with selected retunes toward the March 23 pre-TTT defaults.

Target feature set:

- 11 layers, 512 width, 8 heads, 4 KV heads
- 3x MLP with `LeakyReLU(0.5)^2`
- BigramHash 1536
- XSA on the last 4 layers
- partial RoPE 16/64
- VE128 on layers 9 and 10
- LN scale
- EMA(0.997) + tight SWA every 50 steps when LR scale < 0.2
- warmdown 3500
- seq_len 2048
- train batch 786,432 tokens
- Muon and Adam weight decay 0.04
- sliding-window eval stride 64
- GPTQ-lite int6 export with the five March 22 clip percentiles
- lzma artifact codec

Checkpoints:

- base sanity target: a single-seed run should stay near the March 22 baseline behavior (~1.1228 val_bpb)
- stretch target after the March 23 retunes are fully stacked: approach the March 23 pre-TTT reference (~1.1218 val_bpb)

### Phase 2 - Int4-QAT Ablation On The Frontier Stack

Port the new int4 late-onset Hadamard/trust-gradient QAT onto the parity stack.

Rules:

- support `QAT_BITS=4` only
- require power-of-two `QAT_BLOCK_SIZE`
- keep fail-fast validation
- preserve the existing `QAT_BITS`, `QAT_ONSET_SCALE`, and `QAT_BLOCK_SIZE` semantics
- default `QAT_BITS=0`
- default onset on the frontier stack: `0.15`

Experiment matrix:

- `QAT=off` with `QAT_BITS=0 QAT_ENABLED=0 LATE_QAT_THRESHOLD=0`
- legacy late int6 fake-quant with `QAT_BITS=0 QAT_ENABLED=0 LATE_QAT_THRESHOLD=0.15`
- new int4 late-onset QAT
- onset sweep at `0.10`, `0.15`, `0.20`, `0.30`

Checkpoint: int4 QAT should either improve mean pre-TTT BPB by at least 0.0005 across 3 seeds or materially improve artifact budget enough to justify a scale-up follow-up.

### Phase 3 - Int4 Export And Larger Models

Only proceed if Phase 2 is promising.

- add an int4 export path alongside the int6 GPTQ-lite path
- compare artifact size and roundtrip BPB for int6 vs int4
- scale depth before width by default, keeping width 512 first because it is the least disruptive change to the established stack

### Phase 4 - Legal TTT

After pre-TTT parity and int4-QAT results are stable:

- add legal score-first TTT as a separate layer
- compare post-TTT results to the March 23 record
- do not mix TTT into the first catch-up milestone

## Interfaces And Defaults

- Keep the top-level `train_gpt.py` config surface mostly stable.
- Standardize frontier-only experiments in the new `records/` path around explicit env vars for:
  - activation choice
  - XSA depth
  - RoPE dims
  - VE layers and dim
  - EMA and SWA toggles
  - artifact quantizer and export mode
  - eval stride
  - QAT onset and bit settings
- Treat the March 22 and March 23 record READMEs as the source-of-truth spec for competitive hyperparameters and artifact behavior unless a newer upstream record supersedes them.

## Near-Term Next Steps

1. Run `py_compile` and import-level smoke checks on the new frontier scaffold
2. Run 1-GPU CUDA smoke jobs for `QAT=off` and `QAT_BITS=4`
3. Compare the explicit no-QAT control against the March 22 sanity target first, then the March 23 stretch target
4. Run the int4-QAT onset sweep on the same stack
5. Only then decide whether to invest in int4 export, larger models, or legal TTT
