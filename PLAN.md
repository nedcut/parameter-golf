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
- [x] Re-target the scaffold from March 23 parity toward March 25 parity
- [x] 4-GPU no-QAT upgraded control on the scaffold (frontier4-xsa11-wd4000-noqat, completed 2026-04-01)
  - post-EMA fixed: 1.1171 | float sliding: 1.0935 | int6 fixed: 1.1256 | int6 sliding: 1.1019 | size: 15.62 MB
  - gain vs prior control: `-0.0016` post-EMA fixed | `-0.0025` int6 fixed | `-0.0026` int6 sliding | size also improved
  - new baseline keeps `XSA_LAST_N=11` and `WARMDOWN_ITERS=4000`; biggest March 25 gaps now are BigramHash `3072x112`, GPTQ with AR self-gen calibration, and Parameter Banking + Parallel Muon
- [x] 4-GPU March 25 stack reproduced locally with the full feature set (pg-march25-frontier-4gpu-54842, completed 2026-04-02)
  - generous proxy `11.72`, seed `314`: `7217` steps | float fixed `1.13538259` | float sliding `1.11188900` | int6 sliding `1.11597348` | size `15,864,338`
  - this confirmed the local stack and feature surface, but the proxy was slightly optimistic relative to the original H100 step budget
- [x] 4-GPU tight-proxy matched-budget replication on the March 25 stack (pg-march25-frontier-4gpu-54843/54844, completed 2026-04-02)
  - proxy `11.25`, seeds `314` / `42`: `6932` / `6930` steps vs March 25 record `6927` / `6922`
  - float fixed `1.13584100` / `1.13587012` | float sliding `1.11234307` / `1.11233735`
  - int6 sliding `1.11648884` / `1.11637954` | sizes `15,838,906` / `15,858,893`
  - matched-proxy 2-seed mean `1.11643419` is about `+0.00170` BPB behind the March 25 record mean `1.11473509`
  - seed variance is tiny (`0.00010930` BPB), so the remaining gap looks systematic; export quality is now the leading suspect
- [x] First matched-proxy no-QAT ablation completed on seed `314` (pg-march25-frontier-4gpu-54846, completed 2026-04-03)
  - `6924` steps | float sliding `1.11201207` | int6 sliding `1.11608931` | size `15,989,478`
  - gain vs clean late-QAT seed-314 baseline `54843`: `-0.00033100` float sliding and `-0.00039953` int6 sliding
  - tradeoff: about `+150 KB` artifact growth
- [x] First AR self-gen calibration sweep completed on seed `314` (pg-march25-frontier-4gpu-54848/54849/54850, completed 2026-04-03)
  - `temp=0.9` was the only promising export tweak: int6 sliding `1.11611186`
  - `temp=0.7` regressed to `1.11682006`
  - `seqs=96` regressed badly to `1.11780588` and should not be prioritized
- [x] Clean seed-42 no-QAT rerun completed at a normal matched-proxy pace (pg-march25-frontier-4gpu-54857, completed 2026-04-03)
  - `6924` steps | float sliding `1.11210100` | int6 sliding `1.11617889` | size `15,858,766`
  - gain vs matched seed-42 late-QAT baseline `54844`: `-0.00023635` int6 sliding
  - together with seed `314`, this gives a clean two-seed no-QAT mean of `1.11613410`, about `-0.00030009` better than the late-QAT matched baseline mean `1.11643419`
- [x] Combo test `no-QAT + GPTQ_AR_CALIB_TEMP=0.9` checked on both seeds (pg-march25-frontier-4gpu-54855/54856, completed 2026-04-03)
  - seed `314` run `54855` stopped at only `6745` steps and regressed badly to `1.11797646`; treat as runtime-contaminated rather than a clean verdict
  - seed `42` run `54856` was clean at `6927` steps and landed at `1.11640480`, effectively flat vs the seed-42 late-QAT baseline `1.11637954`
  - current read: `temp=0.9` is still a plausible export-only tweak, but it is not a reliable additive gain on top of no-QAT
- [x] Milder late-QAT follow-up checked and deprioritized (pg-march25-frontier-4gpu-54861, completed 2026-04-03)
  - `LATE_QAT_THRESHOLD=0.10` with `GPTQ_AR_CALIB_TEMP=0.9` reached `6808` steps and regressed to int6 sliding `1.11751938`
  - size also regressed to `16,048,254` bytes, so this setting should not be pursued further
- [x] March 25 matched-proxy closeout finished after reruns and calibration-seed checks (pg-march25-frontier-4gpu-54865/54866/54867/54868, completed 2026-04-04)
  - clean rerun `54866` (`seed=314`, no-QAT, `temp=0.9`) reached `6932` steps and improved to float sliding `1.11196138` | int6 sliding `1.11587877` | size `15,856,990`
  - overlapping rerun `54865` only reached `5953` steps and should be treated as runtime-contaminated, not as a recipe datapoint
  - calibration-seed checks were not promising: clean `54868` (`GPTQ_AR_CALIB_SEED=42`) regressed to int6 sliding `1.11628764`, while `54867` was both slower and worse
  - final March 25 read: freeze the local parity base at `H100_EQUIV_MULTIPLIER=11.25` and `LATE_QAT_THRESHOLD=0`; treat `GPTQ_AR_CALIB_TEMP=0.9` as optional, but stop broader export and legacy late-QAT sweeps
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
  - AR self-gen GPTQ calibration count / seq len / temperature / batch / seed
  - eval stride
  - QAT onset and bit settings
- Treat the March 22, March 23, and March 25 record READMEs as the source-of-truth spec for competitive hyperparameters and artifact behavior, with March 25 taking precedence for the current parity target.

## Near-Term Next Steps

1. Freeze the March 25 local proxy baseline at `H100_EQUIV_MULTIPLIER=11.25` with `LATE_QAT_THRESHOLD=0`; use `GPTQ_AR_CALIB_TEMP=0.8` as the conservative default and treat `0.9` as optional rather than promoted
2. Finish the local matched-proxy control as a real 3-seed no-QAT baseline by adding `seed=999`
3. Port int4 late-onset Hadamard / trust-gradient QAT onto that exact frozen no-QAT stack before doing any more broad March 25 micro-sweeps
4. Start the int4 check with a narrow seed-314 matrix: control, onset `0.15`, and onset `0.20`
5. Do not spend more time on legacy late-QAT onset sweeps, `GPTQ_AR_CALIB_SEQS > 64`, or calibration-seed sweeps unless the int4 work specifically exposes a new export bottleneck
