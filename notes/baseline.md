# Baseline notes

## What `train_gpt.py` actually optimizes

The challenge metric is the post-quantization roundtrip `val_bpb`, not just raw validation loss.

Important consequences:
- model quality matters
- compressed artifact size matters
- quantization friendliness matters
- tiny code changes can matter because code bytes count too

## Main knobs discovered in `train_gpt.py`

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

### Training schedule
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

### Quantization / compression
- per-row int8 for large matrices
- per-tensor int8 for vectors/scalars
- small/control tensors preserved in fp16/fp32 passthrough
- final artifact is `torch.save(...)` of quantized state, then `zlib.compress(..., level=9)`
- headline metric is logged as:
  - `final_int8_zlib_roundtrip val_loss:... val_bpb:...`
  - `final_int8_zlib_roundtrip_exact val_loss:... val_bpb:...`

## High-probability levers

1. **Architecture frontier search**
   - width/depth/head/GQA tradeoffs likely move both quality and compressibility
2. **Schedule tuning**
   - especially learning rates and warmdown under the wallclock cap
3. **Compressibility-aware tweaks**
   - weight sharing / tying / lower-entropy parameter distributions

## Key non-obvious detail

The script requires CUDA and assumes `WORLD_SIZE` divides 8 so gradient accumulation stays integral.
That means even single-GPU smoke tests should use the CUDA path and preferably `torchrun --standalone --nproc_per_node=1`.
