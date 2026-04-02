# March 25 Frontier Proxy Run Notes

Date: 2026-04-02
Log: `slurm/output/pg-march25-frontier-4gpu-54842.out`
Stack: `records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072`

## Summary

This run cleanly reproduced the March 25 no-TTT frontier stack on 4 GPUs with the local H100 proxy enabled.

Key metrics:

- step cap reached at `7217`
- post-EMA fixed `val_bpb`: `1.13538259`
- float sliding `val_bpb`: `1.11188900`
- int6 fixed `val_bpb`: `1.13953472`
- int6 sliding `val_bpb`: `1.11597348`
- total submission size: `15,864,338` bytes

## What Looks Good

- Training matched the upstream seed-314 pre-quant behavior almost exactly.
- The full March 25 stack is working locally: XSA-all, BigramHash `3072x112`, warmdown `4000`, Parallel Muon + Parameter Banking, AR self-gen GPTQ, and selective pruning logic all executed as expected.
- Memory headroom was comfortable on this setup: about `23.1 GiB` allocated at peak.

## Main Caveat

The current proxy multiplier was slightly optimistic for this run.

- This run used `H100_EQUIV_MULTIPLIER=11.72`.
- With the observed `974.54 ms/step`, that implies about `83.15 ms/step` H100-equivalent.
- The March 25 reference reports about `86.6 ms/step`.
- A multiplier near `11.25` would have landed almost exactly on the reference step budget of about `6927`.

So this result is best treated as "near-parity under a mildly generous proxy cap," not yet the final apples-to-apples parity check.

## Interpretation

- The float model is already strong enough to beat the public `1.1147` target before quantization on sliding eval.
- The remaining gap is mostly export-side: quantization cost was about `+0.0041` BPB on both fixed and sliding eval.
- Artifact headroom is tight: only about `35.7 KB` under the `15.9 MB` target.
- Because the March 25 script does not currently expose AR self-gen temperature or sequence-count knobs as env vars, the cleanest immediate experiments are tighter-proxy replications and schedule ablations that already exist in the config surface.

## Next Three Runs

### 1. Tight-proxy parity check, seed 314

```bash
sbatch --export=ALL,RUN_ID=march25-proxy1125-s314,H100_PROXY=1,H100_EQUIV_MULTIPLIER=11.25,SEED=314 slurm/train_march25_frontier_4gpu.sbatch
```

Purpose: repeat the strongest seed with a proxy budget that matches the original March 25 step count more closely.

### 2. Tight-proxy replication, seed 42

```bash
sbatch --export=ALL,RUN_ID=march25-proxy1125-s42,H100_PROXY=1,H100_EQUIV_MULTIPLIER=11.25,SEED=42 slurm/train_march25_frontier_4gpu.sbatch
```

Purpose: start a matched local multi-seed picture instead of over-indexing on one favorable seed.

### 3. Tight-proxy late-QAT ablation, seed 314

```bash
sbatch --export=ALL,RUN_ID=march25-proxy1125-s314-noqat,H100_PROXY=1,H100_EQUIV_MULTIPLIER=11.25,SEED=314,LATE_QAT_THRESHOLD=0 slurm/train_march25_frontier_4gpu.sbatch
```

Purpose: measure how much of the March 25 performance on this hardware still depends on late int6 fake-quant once the proxy budget is matched.

## Decision Rule After These

- If the `11.25` proxy runs stay close to `1.115` quantized sliding BPB, treat March 25 parity as effectively reproduced locally.
- If disabling late QAT hurts clearly, keep it in the parity stack and port the new int4 QAT onto this exact stack rather than waiting on more scaffold catch-up.
- If the no-QAT ablation is neutral or better, the next work should focus on the export path and quantization stability rather than training-time QAT.
