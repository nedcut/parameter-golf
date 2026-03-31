# Starter Trainer vs Frontier Stack

## What The Top-Level `train_gpt.py` Optimizes

The challenge metric is post-quantization roundtrip `val_bpb`, not just raw validation loss.

Important consequences:

- model quality matters
- compressed artifact size matters
- quantization friendliness matters
- code size still matters

The top-level trainer is still a good starting point, but it is intentionally not the current frontier stack.

## Main Knobs In The Starter Trainer

### Architecture

- `VOCAB_SIZE` (default 1024)
- `NUM_LAYERS` (default 9)
- `MODEL_DIM` (default 512)
- `NUM_HEADS` (default 8)
- `NUM_KV_HEADS` (default 4)
- `MLP_MULT` (default 2)
- `TIE_EMBEDDINGS` (default 1)
- `ROPE_BASE`
- `LOGIT_SOFTCAP`
- `QK_GAIN_INIT`

### Training Schedule

- `ITERATIONS` (default 20000)
- `WARMUP_STEPS` (default 20)
- `WARMDOWN_ITERS` (default 1200)
- `MAX_WALLCLOCK_SECONDS` (default 600)
- `TRAIN_BATCH_TOKENS` (default 524288)
- `TRAIN_SEQ_LEN` (default 1024)
- `VAL_LOSS_EVERY`
- `TRAIN_LOG_EVERY`

### Optimizer

- `EMBED_LR`
- `HEAD_LR`
- `TIED_EMBED_LR`
- `MATRIX_LR`
- `SCALAR_LR`
- `MUON_MOMENTUM`
- `MUON_BACKEND_STEPS`
- `MUON_MOMENTUM_WARMUP_START`
- `MUON_MOMENTUM_WARMUP_STEPS`
- `BETA1`, `BETA2`, `ADAM_EPS`
- `GRAD_CLIP_NORM`

### Quantization / Compression

- final artifact: int8 + zlib roundtrip
- new training-time path: `QAT_BITS=4` late-onset Hadamard/trust-gradient fake quant
- QAT controls:
  - `QAT_BITS`
  - `QAT_ONSET_SCALE`
  - `QAT_BLOCK_SIZE`

## Frontier Snapshot As Of March 26, 2026

Current public 10-minute SOTA:

- `1.1194 val_bpb`
- March 23 record: `LeakyReLU^2 + Legal Score-First TTT + Parallel Muon`

Current stretch target once the March 23 retunes are fully in place:

- about `1.1218 val_bpb` from the March 23 stack before TTT

Nearest codebase checkpoint to build from:

- March 22 `11L EMA + GPTQ-lite + warmdown3500 + QAT@0.15`

Important nuance:

- the current frontier scaffold starts from the March 22 codebase
- it already retunes some settings toward March 23, including `LeakyReLU(0.5)^2`, `BIGRAM_VOCAB_SIZE=1536`, and `lzma`
- it is not yet the full March 23 stack because Parameter Banking + Parallel Muon and legal TTT are still absent

## Starter Trainer vs Frontier Stack

| Area | Starter `train_gpt.py` | Frontier stack | Where it should live |
|------|------------------------|----------------|----------------------|
| Core role | readable baseline | competitive record path | keep separate |
| MLP activation | `relu^2` | `LeakyReLU(0.5)^2` | competitive `records/` path |
| MLP width | 2x | 3x | competitive `records/` path |
| Sequence length | 1024 | 2048 | competitive `records/` path |
| RoPE | full | partial 16/64 | competitive `records/` path |
| Attention tweak | none | XSA on last 4 layers | competitive `records/` path |
| Token identity features | none | BigramHash + VE128 | competitive `records/` path |
| Averaging | none | EMA + tight SWA | competitive `records/` path |
| Export | int8 + zlib | GPTQ-lite int6 + lzma | competitive `records/` path |
| Eval | fixed-window | sliding-window stride 64 | competitive `records/` path |
| Weight decay | none | Muon/Adam WD 0.04 | competitive `records/` path |
| Optimizer systems | classic Muon | Parameter Banking + Parallel Muon | later competitive `records/` path |
| New idea | int4 Hadamard/trust-gradient QAT | stack on frontier recipe | shared logic plus competitive `records/` path |

## Working Decision

Do not bloat the beginner-oriented top-level trainer with every frontier-only knob.

Instead:

- keep `train_gpt.py` usable as a starter trainer
- keep genuinely reusable logic small and isolated
- land competitive catch-up work in a new `records/track_10min_16mb/...` training path

Current local scaffold for that work:

- `records/track_10min_16mb/2026-03-26_11L_PreTTT_Frontier_Int4QAT`

## High-Probability Levers From Here

1. Match the March 22 sanity target first, then close the gap toward the March 23 pre-TTT stretch target
2. Ablate int4 late-onset QAT on that stronger stack
3. Only then explore int4 export and larger models
4. Add legal score-first TTT after pre-TTT parity is stable

## Non-Obvious Operational Detail

The CUDA path still assumes `WORLD_SIZE` divides 8 so gradient accumulation stays integral.

That means even single-GPU smoke tests should use:

- `torchrun --standalone --nproc_per_node=1 ...`
