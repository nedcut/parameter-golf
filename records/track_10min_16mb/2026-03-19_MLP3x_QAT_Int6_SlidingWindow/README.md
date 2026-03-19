# MLP 3x + QAT Int6 + FP16 Embed + Sliding Window Eval

## Summary

Wider MLP (3x expansion) with int6 quantization-aware training, fp16 tied embedding export, and sliding window evaluation. This achieves **val_bpb = 1.1652** (mean across 5 seeds), a **0.1053 nat improvement** over the naive baseline.

### Key Changes from Baseline

1. **Wider MLP (MLP_MULT=3)** — 3x expansion (hidden=1536 vs 1024), more capacity per layer
2. **Quantization-Aware Training (QAT)** — STE fake-quantize in CastedLinear simulates int6 noise during training, reducing post-quant degradation from ~0.01 to ~0.001 BPB
3. **Int6 quantization on all block weights** (layers 0-8) — rounds to multiples of 4 in int8, giving 64 effective levels. Compresses much better with zlib.
4. **FP16 tied embedding export** — keeps `tok_emb.weight` in fp16 instead of int8
5. **Lower learning rates**: MATRIX_LR=0.02, SCALAR_LR=0.02, TIED_EMBED_LR=0.03
6. **Sliding window evaluation (stride=64)** — overlapping eval windows give each scored token 960 tokens of context, ~0.033 BPB free improvement

### Architecture

- 9 transformer blocks, 512 model dim, 8 attention heads, 4 KV heads
- GQA attention with RoPE, ReLU² MLP (**3x** expansion)
- Tied embeddings with 1024 BPE vocabulary
- U-Net skip connections (4 encoder + 5 decoder layers)
- 21.8M parameters, 15.64MB compressed artifact

### Export Strategy

| Component | Precision | Rationale |
|---|---|---|
| tok_emb.weight (tied) | fp16 | Most sensitive — serves as both embedding and output head |
| All block weights (layers 0-8) | int6 (64 levels) | QAT makes model robust to int6 noise |
| Small tensors / scalars | fp16 passthrough | Standard baseline behavior |

## Multi-Seed Results (5 seeds, p << 0.001)

| Seed | slide_loss (nats) | slide_bpb | rt_bpb | Artifact |
|---|---|---|---|---|
| 1337 | 1.96976839 | 1.16660881 | 1.19979073 | 15,637,847 |
| 42 | 1.96607538 | 1.16442160 | 1.19766590 | 15,643,141 |
| 123 | 1.96296420 | 1.16257898 | 1.19617553 | 15,637,116 |
| 7 | 1.97030598 | 1.16692720 | 1.20001881 | 15,643,756 |
| 2024 | 1.96775823 | 1.16541828 | 1.19865532 | 15,636,562 |
| **Mean** | **1.96737444** | **1.16519097** | **1.19846126** | **15,639,684** |
| **Std** | **0.00298371** | | | |

- **Mean improvement: 0.1053 nats** (21× the 0.005 threshold)
- **t-statistic: 78.93** (df=4, one-sided t critical for p<0.01 = 3.747)
- **p << 0.001**
- All 5 artifacts under 16MB
- Sliding window eval takes ~73s on 8xH100 (well under 10-min eval budget)

## Hardware

All runs on 8×H100 SXM (RunPod). Each run completes in ~600s training (wallclock cap) + ~73s sliding window eval, reaching ~12,330 training steps at ~48.7ms/step.

## How to Run

```bash
RUN_ID=submission \
SEED=1337 \
NUM_LAYERS=9 \
MLP_MULT=3 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
TIED_EMBED_LR=0.03 \
FP16_EMBED_EXPORT=1 \
INT6_LAYER_START=0 \
INT6_LAYER_END=8 \
QAT_ENABLED=1 \
QAT_INT6=1 \
EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
