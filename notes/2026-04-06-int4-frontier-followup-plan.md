# Int4 Frontier Follow-Up Plan

Date: 2026-04-06
Stack: `records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072`

## Logs Reviewed

- `slurm/output/pg-march25-frontier-4gpu-54846.out` - seed `314` no-QAT control, `temp=0.8`
- `slurm/output/pg-march25-frontier-4gpu-54857.out` - seed `42` no-QAT control, `temp=0.8`
- `slurm/output/pg-march25-frontier-4gpu-54892.out` - int4 onset `0.15`, seed `314`
- `slurm/output/pg-march25-frontier-4gpu-54893.out` - int4 onset `0.20`, seed `314`
- `slurm/output/pg-march25-frontier-4gpu-54928.out` - int4 onset `0.20` rerun, seed `314`
- `slurm/output/pg-march25-frontier-4gpu-54929.out` - int4 onset `0.10`, seed `314`
- `slurm/output/pg-march25-frontier-4gpu-54930.out` - int4 onset `0.30`, seed `314`

## Seed-314 Comparison Table

Baseline for the int4 sweep is the clean no-QAT seed-314 control `54846`:

- steps: `6924`
- float sliding: `1.11201207`
- int6 sliding: `1.11608931`
- size: `15,989,478` bytes

| Log | Onset | Steps | Float sliding | Delta vs control | Int6 sliding | Delta vs control | Size | Delta size | Read |
|-----|-------|-------|---------------|------------------|--------------|------------------|------|------------|------|
| `54929` | `0.10` | `6760` | `1.11416680` | `+0.00215473` | `1.11788573` | `+0.00179642` | `15,997,075` | `+7,597` | dead |
| `54892` | `0.15` | `6908` | `1.11279201` | `+0.00077994` | `1.11688663` | `+0.00079732` | `15,849,639` | `-139,839` | worse; size win not enough |
| `54893` | `0.20` | `7219` | `1.11235353` | `+0.00034146` | `1.11594526` | `-0.00014405` | `15,980,703` | `-8,775` | looked promising once |
| `54928` | `0.20` | `6960` | `1.11394028` | `+0.00192821` | `1.11767128` | `+0.00158197` | `15,958,051` | `-31,427` | non-repro rerun |
| `54930` | `0.30` | `6908` | `1.11250945` | `+0.00049738` | `1.11659229` | `+0.00050298` | `15,861,167` | `-128,311` | only borderline survivor |

## Read

### 1. Onset `0.10` is out

`54929` is clearly worse on both float and quantized eval, and it does not buy meaningful size relief. No more time should go here.

### 2. Onset `0.15` is out

`54892` is a stable regression. The roughly `140 KB` size reduction is real, but not strong enough to justify nearly `+0.0008` BPB on both float and quantized sliding eval.

### 3. Onset `0.20` looks too unstable to keep investing in

`54893` briefly looked good on quantized eval, but the rerun `54928` did not reproduce it. The pair now reads like noise or runtime sensitivity, not a recipe we can trust. Unless a later result forces us back here, stop rerunning `0.20`.

### 4. Onset `0.30` is the only setting still worth one more check

`54930` is not a win, but it is close enough to the no-QAT control to justify one cross-seed test:

- float sliding regressed by about `+0.00050`
- int6 sliding regressed by about `+0.00050`
- submission size improved by about `128 KB`

That is still below the stated success bar, but it is the only onset left that looks like a plausible tradeoff instead of a clear miss.

## Adapted Plan For The Rest Of The Week

### Tuesday, 2026-04-07

Run exactly one new int4 job:

```bash
sbatch --export=ALL,RUN_ID=march25-proxy1125-s42-int4o30,H100_PROXY=1,H100_EQUIV_MULTIPLIER=11.25,SEED=42,QAT_BITS=4,QAT_ONSET_SCALE=0.30,QAT_BLOCK_SIZE=128,GPTQ_AR_CALIB_TEMP=0.8 slurm/train_march25_frontier_4gpu.sbatch
```

Reason: `0.30` is the only onset that still has a case.

### Wednesday, 2026-04-08

Decision gate from the seed-42 run:

- if seed-42 `0.30` is worse than the clean seed-42 no-QAT control `54857` by more than about `0.0005` BPB, close the int4 phase
- if seed-42 `0.30` is roughly flat and preserves the size reduction, promote it to one final 3-seed check

### Thursday, 2026-04-09

Only if seed-42 `0.30` survives:

```bash
sbatch --export=ALL,RUN_ID=march25-proxy1125-s999-int4o30,H100_PROXY=1,H100_EQUIV_MULTIPLIER=11.25,SEED=999,QAT_BITS=4,QAT_ONSET_SCALE=0.30,QAT_BLOCK_SIZE=128,GPTQ_AR_CALIB_TEMP=0.8 slurm/train_march25_frontier_4gpu.sbatch
```

Optional bookkeeping if we need a perfectly matched 3-seed control:

```bash
sbatch --export=ALL,RUN_ID=march25-proxy1125-s999-noqat-apr06b,H100_PROXY=1,H100_EQUIV_MULTIPLIER=11.25,SEED=999,LATE_QAT_THRESHOLD=0,GPTQ_AR_CALIB_TEMP=0.8 slurm/train_march25_frontier_4gpu.sbatch
```

### Friday, 2026-04-10

Make the closeout call:

- keep int4 alive only if the best surviving onset clears the project bar of at least `0.0005` mean BPB improvement or creates clearly useful artifact headroom
- otherwise mark the int4 onset sweep complete and move on

## Things Not To Do This Week

- no more reruns of onset `0.10`
- no more reruns of onset `0.15`
- no more reruns of onset `0.20` unless a new signal appears
- no block-size sweep
- no AttnRes work yet
- no more calibration-temperature or calibration-seed sweeps for the int4 triage path

## Bottom Line

The fresh runs simplify the story a lot:

- `0.10` is dead
- `0.15` is dead
- `0.20` is not reproducible enough
- `0.30` is the only onset still worth one more seed

If seed-42 `0.30` does not hold up, the clean move is to close the int4 onset sweep and stop spending time on it this week.
