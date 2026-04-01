# Parameter Golf Plan

Goal: bring this repo to current no-TTT frontier parity, then test int4 late-onset Hadamard/trust-gradient QAT on that stronger stack.

## Context

- **Challenge**: best LM in a 16MB artifact, trained in <=10 minutes on 8xH100s, scored on FineWeb val BPB
- **Deadline**: April 30, 2026
- **Current 10-minute SOTA**: 1.1147 val_bpb (`AR Self-Gen GPTQ + XSA-all + BigramHash 3072x112`, 3-seed mean, no TTT)
- **Current no-TTT frontier target**: ~1.1147 val_bpb from the March 25 record stack
- **Older catch-up waypoint**: ~1.1218 val_bpb from the March 23 record stack before legal TTT
- **Nearest reproducible codebase checkpoint**: March 22 `11L EMA + GPTQ-lite + warmdown3500 + QAT@0.15` record stack

Late QAT is already part of the public frontier. Our new int4 Hadamard/trust-gradient QAT should be treated as a new ingredient to stack on top of the strongest no-TTT recipe we can reproduce, not as the only missing idea.

## Gap Analysis

The top-level `train_gpt.py` remains a strong starter trainer, but it still lacks several frontier ingredients:

| Area | Starter trainer | Frontier stack | Where to implement |
|------|-----------------|----------------|--------------------|
| MLP activation | `relu^2` | `LeakyReLU(0.5)^2` | competitive `records/` path |
| MLP width | 2x | 3x | competitive `records/` path |
| Context | 1024 train/eval | 2048 train/eval | competitive `records/` path |
| Positional setup | full RoPE | partial RoPE 16/64 | competitive `records/` path |
| Attention tweak | none | XSA on all 11 layers | competitive `records/` path |
| Token identity features | none | BigramHash + VE128 | competitive `records/` path |
| Weight averaging | none | EMA + tight SWA | competitive `records/` path |
| Export path | int8 + zlib | full-Hessian GPTQ + self-generated calibration | reusable quant helpers ok, full path in `records/` |
| Weight decay | none | Muon/Adam weight decay 0.04 | competitive `records/` path |
| Eval mode | fixed-window | sliding-window stride 64 | competitive `records/` path |
| Optimizer systems | classic Muon | Parameter Banking + Parallel Muon | later competitive `records/` path |
| Current frontier extras | none | BigramHash `3072x112`, selective pruning, AR self-gen calibration | competitive `records/` path |
| New QAT idea | int4 Hadamard/trust-gradient in starter script | not yet stacked on frontier | shared logic plus competitive `records/` path |

Decision: keep the beginner-oriented top-level trainer mostly stable. Competitive catch-up work should land in a new `records/track_10min_16mb/...` path, except for small reusable utilities that are genuinely shared.

## Current State

- [x] Leaderboard snapshot verified against upstream as of March 31, 2026
- [x] New int4 late-onset Hadamard/trust-gradient QAT implemented and hardened in top-level `train_gpt.py`
- [x] Internal docs updated to reflect the real frontier
- [x] New pre-TTT frontier scaffold created under `records/track_10min_16mb/2026-03-26_11L_PreTTT_Frontier_Int4QAT`
- [x] 1-GPU CUDA smoke matrix on the new frontier scaffold
- [x] 4-GPU no-QAT control on the scaffold (frontier4-a-noqat-s1337, completed 2026-03-31)
  - post-EMA val_bpb: 1.1187 | int6 roundtrip: 1.1281 | int6 sliding-window: 1.1045 | size: 15.75 MB
  - XSA last_4 only, warmdown 3500 — March 25 gaps (XSA-all, BigramHash 3072x112, warmdown 4000) still to close
- [ ] Re-target the scaffold from March 23 parity toward March 25 parity
- [ ] Pre-TTT parity check against the March 25 no-TTT stack
- [ ] Int4-QAT ablation on the parity stack
- [ ] Int4 export / larger-model follow-up if int4 QAT shows value
- [ ] Optional legal score-first TTT only after no-TTT parity is stable

## Roadmap

### Phase 1 - March 25 No-TTT Frontier Parity

Build and validate a reproducible no-TTT frontier path in a new record-style folder based on the March 22 competitive script, then close the largest gaps to the March 25 upstream record.

Target feature set:

- 11 layers, 512 width, 8 heads, 4 KV heads
- 3x MLP with `LeakyReLU(0.5)^2`
- XSA on all 11 layers
- BigramHash `3072x112`
- partial RoPE 16/64
- VE128 on layers 9 and 10
- LN scale
- EMA(0.997) + tight SWA every 50 steps when LR scale < 0.2
- warmdown 4000
- seq_len 2048
- train batch 786,432 tokens
- Muon and Adam weight decay 0.04
- sliding-window eval stride 64
- Parameter Banking + Parallel Muon
- full-Hessian GPTQ with AR self-generated calibration
- selective pruning of `+/-1` values by reconstruction error
- lzma artifact codec

Checkpoints:

- base sanity target: a single-seed run should stay near the March 22 baseline behavior (~1.1228 val_bpb)
- intermediate waypoint after the early retunes: approach the March 23 pre-TTT reference (~1.1218 val_bpb)
- parity target after the March 25 ingredients are stacked: approach the March 25 no-TTT reference (~1.1147 val_bpb)

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

### Phase 4 - Optional Legal TTT Layer

Only revisit this if it still looks promising after no-TTT parity and int4-QAT results are stable:

- add legal score-first TTT as a separate layer
- compare post-TTT results against the best no-TTT stack rather than treating it as the default target
- do not mix TTT into the first catch-up milestone

### Phase 5 - Speculative Later Work

These are worth tracking, but they should not preempt March 25 parity:

- Gram-Newton-Schulz for H100-oriented Muon speedups, ideally paired with Parameter Banking + Parallel Muon
- end-to-end lower-bit artifact-first ideas inspired by Bonsai / BitNet-style efficiency work
- larger-model follow-ups only after the current 11L stack is competitive again

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
- Treat the March 22, March 23, and March 25 record READMEs as the source-of-truth spec for competitive hyperparameters and artifact behavior, with March 25 taking precedence for the current parity target.

## Near-Term Next Steps

1. Let the current 4-GPU no-QAT control finish and use it as the reference run for the scaffold
2. Port the highest-signal March 25 deltas first: XSA-all, BigramHash `3072x112`, and warmdown 4000
3. Re-run a matched no-QAT smoke and then a 4-GPU control on that stronger stack
4. Port Parameter Banking + Parallel Muon once the architectural deltas are in place
5. Only after that, revisit int4-QAT on the stronger stack and judge whether export work is justified
