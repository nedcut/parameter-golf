# Parameter Golf Plan

Goal: push past the current SOTA (`1.1194 val_bpb`) via quantization-aware training, enabling either better post-quantization quality at the same model size or a larger model in the same 16MB budget.

## Context

- **Challenge**: Best LM in a 16MB artifact, trained in ≤10 min on 8×H100s, scored on FineWeb val BPB
- **Deadline**: April 30, 2026
- **Current SOTA**: 1.1194 val_bpb (LeakyReLU² + Legal TTT + Parallel Muon, 3-seed mean)
- **Naive baseline**: 1.2244 val_bpb (9L 512d, int8+zlib)
- **Unlimited-compute frontier**: 1.1239 val_bpb (1-bit, 106M params, 2.15 hours)

The leaderboard has been heavily optimized along architecture (11L, XSA, Partial RoPE, BigramHash), training (Muon, EMA, warmdown), and post-training quantization (GPTQ-lite int6). The remaining frontier is **training-time quantization awareness** — making the model learn weight distributions that compress better.

## Current state

- [x] Codebase understood — `train_gpt.py` fully read, all knobs documented in `notes/baseline.md`
- [x] Leaderboard analyzed — top 5 submissions cross-referenced, ablation tables reviewed
- [x] **Int4 late-onset QAT implemented** in `train_gpt.py` (Hadamard + trust gradient, QuEST-inspired)
- [ ] QAT smoke test on GPU (needs CUDA)
- [ ] Int4 export path (currently QAT trains int4-friendly weights, export still int8+zlib)
- [ ] First A/B comparison: baseline vs QAT on identical architecture

## Strategy

Two parallel tracks, both building on the QAT infrastructure:

### Track A — Better quantization at same model size
- Train with `QAT_BITS=4 QAT_ONSET_SCALE=0.2` on the current 11L/512d architecture
- Export at int8+zlib (unchanged) and compare val_bpb to non-QAT baseline
- Hypothesis: QAT-trained weights have tighter distributions → lower post-quantization error → better BPB
- Low risk, easy to A/B test

### Track B — Larger model via int4 export
- Add int4+Hadamard export path alongside existing int8
- Scale up architecture (more layers or wider) to fill the freed bytes
- Hypothesis: int4 export of QAT-trained model fits ~50% more params in 16MB
- Higher risk, depends on Track A working first

## Implementation plan

### Phase 1 — Validate QAT (current)

1. **Smoke test** on GPU: run `QAT_BITS=4` for a short run, verify:
   - No crashes or NaN gradients
   - `torch.compile` handles the onset recompile
   - QAT onset logging works
   - Training speed overhead is acceptable (<15%)
2. **A/B run**: same architecture, same seed, ±QAT, compare final int8+zlib roundtrip BPB
3. **Onset sweep**: try `QAT_ONSET_SCALE` in {0.1, 0.2, 0.3, 0.5} to find the compute-optimal onset point

### Phase 2 — Int4 export path

1. Add `quantize_state_dict_int4()` with:
   - Hadamard pre-rotation of weight matrices before quantizing
   - Per-row clip search (same as GPTQ-lite but at 4-bit range)
   - lzma compression
2. Measure artifact size vs int6 and int8 at the same model size
3. If artifact is meaningfully smaller, proceed to Phase 3

### Phase 3 — Scale up the model

Candidate architecture changes (pick one or stack):
- 11L → 14L at same width (512d) — more depth
- 512d → 640d at same depth (11L) — more width
- MLP 2× → 3× (if not already at 3×) — wider MLP
- All of the above combined if int4 frees enough bytes

Run 3-seed evaluation on best config.

### Phase 4 — Stack with SOTA techniques

Once the QAT + larger model works, stack on:
- LeakyReLU(0.5)² (known -0.003 BPB)
- XSA on last 4 layers
- EMA + Tight SWA
- Legal TTT
- Parameter Banking + Parallel Muon

This is where we'd create a proper `records/` submission.

## QAT implementation details

Added to `train_gpt.py` (1227 lines, under the 1500 cap):

| Component | Description |
|-----------|-------------|
| `HadamardTrustQuantizer` | Simulates int-N quantization in forward pass with Hadamard pre-rotation and trust-region gradient masking |
| `_build_hadamard_block(128)` | Sylvester-construction normalized Hadamard matrix, H²=I |
| `_hadamard_rotate` | Block-diagonal rotation via reshape+matmul, no full matrix stored |
| `CastedLinear.wq` | Optional quantizer submodule, created when `qat_bits > 0` |
| `GPT.set_qat_enabled()` | Toggles all quantizers on/off |
| Late-onset trigger | In training loop: enables QAT when `lr_scale <= QAT_ONSET_SCALE` |

Env vars: `QAT_BITS` (default 0), `QAT_ONSET_SCALE` (default 0.2), `QAT_BLOCK_SIZE` (default 128)

## Key references

- **QuEST** (arXiv 2502.05003): Hadamard + trust gradient for stable sub-4-bit QAT
- **Compute-Optimal QAT** (arXiv 2509.22935): Late-onset QAT during LR cooldown
- **"Low-Bit Quantization Favors Undertrained LLMs"** (ACL 2025): Aggressive quantization hurts less on undertrained models (relevant since our 10-min runs don't fully converge)

## Near-term next steps

1. Get GPU access (RunPod or similar) for smoke tests
2. Run QAT smoke test
3. Run A/B comparison (±QAT, same architecture)
4. If positive, build int4 export path
5. If still positive, scale up model and run 3-seed eval
