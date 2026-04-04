# March 25 Frontier Proxy Run Notes

Date: 2026-04-02
Stack: `records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072`
Logs:

- `slurm/output/pg-march25-frontier-4gpu-54842.out`
- `slurm/output/pg-march25-frontier-4gpu-54843.out`
- `slurm/output/pg-march25-frontier-4gpu-54844.out`

## Summary

The local 4-GPU March 25 stack is now reproducible under both the earlier generous proxy and a tighter proxy that matches the original step budget almost exactly.

The most important result is the tight-proxy pair:

- seed `314`: `6932` steps, float fixed `1.13584100`, float sliding `1.11234307`, int6 sliding `1.11648884`, size `15,838,906` bytes
- seed `42`: `6930` steps, float fixed `1.13587012`, float sliding `1.11233735`, int6 sliding `1.11637954`, size `15,858,893` bytes
- tight-proxy 2-seed mean: `6931` steps, float sliding `1.11234021`, int6 sliding `1.11643419`

Those two seeds differ by only `0.00010930` BPB on quantized sliding eval, so this now looks like a real local plateau rather than a lucky or unlucky seed.

## Run Table

| Log | Proxy | Seed | Steps | Float fixed | Float sliding | Int6 fixed | Int6 sliding | Size |
|-----|-------|------|-------|-------------|---------------|------------|--------------|------|
| `54842` | `11.72` | `314` | `7217` | `1.13538259` | `1.11188900` | `1.13953472` | `1.11597348` | `15,864,338` |
| `54843` | `11.25` | `314` | `6932` | `1.13584100` | `1.11234307` | `1.14007656` | `1.11648884` | `15,838,906` |
| `54844` | `11.25` | `42` | `6930` | `1.13587012` | `1.11233735` | `1.14000097` | `1.11637954` | `15,858,893` |

## Interpretation

### 1. The `11.25` proxy is the right local baseline

- The original record steps were `6927`, `6922`, and `6917`.
- The tight-proxy local runs landed at `6932` and `6930`, which is close enough that proxy mismatch is no longer the main explanation.
- The earlier `11.72` proxy was optimistic by about `285` steps and improved quantized sliding BPB by about `0.0005`, so it should not be used for parity claims.

### 2. We are close, but not yet at March 25 parity

- Tight-proxy local mean: `1.11643419`
- March 25 record mean: `1.11473509`
- Remaining gap: about `+0.00169910` BPB

This is no longer a "missing major feature" gap. It is the sort of gap that could come from a mix of local hardware/runtime differences, training details, and export quality.

### 3. Export still looks like the biggest lever

- Tight-proxy float sliding mean is `1.11234021`.
- Tight-proxy int6 sliding mean is `1.11643419`.
- That is about `+0.00409` BPB of quantization/export cost.

Late QAT is active only for the final ~`600` steps in these matched runs, and selective pruning is not binding because every run already fits without pruning. That makes export calibration the highest-leverage place to poke next.

## What I Would Change Next

### Priority 1: matched-proxy no-QAT ablation

Run the already-planned no-QAT check at the correct proxy budget before changing anything more structural.

```bash
sbatch --export=ALL,RUN_ID=march25-proxy1125-s314-noqat,H100_PROXY=1,H100_EQUIV_MULTIPLIER=11.25,SEED=314,LATE_QAT_THRESHOLD=0 slurm/train_march25_frontier_4gpu.sbatch
sbatch --export=ALL,RUN_ID=march25-proxy1125-s42-noqat,H100_PROXY=1,H100_EQUIV_MULTIPLIER=11.25,SEED=42,LATE_QAT_THRESHOLD=0 slurm/train_march25_frontier_4gpu.sbatch
```

Reason: if late QAT is neutral here, the main remaining work is export-side, not training-side.

### Priority 2: sweep AR self-gen GPTQ calibration

The frontier script now exposes the AR self-gen calibration knobs as env vars:

- `GPTQ_AR_CALIB_SEQS`
- `GPTQ_AR_CALIB_SEQ_LEN`
- `GPTQ_AR_CALIB_TEMP`
- `GPTQ_AR_CALIB_BATCH_SIZE`
- `GPTQ_AR_CALIB_SEED`

Start with temperature and sequence count on the stronger seed:

```bash
sbatch --export=ALL,RUN_ID=march25-proxy1125-s314-temp07,H100_PROXY=1,H100_EQUIV_MULTIPLIER=11.25,SEED=314,GPTQ_AR_CALIB_TEMP=0.7 slurm/train_march25_frontier_4gpu.sbatch
sbatch --export=ALL,RUN_ID=march25-proxy1125-s314-temp09,H100_PROXY=1,H100_EQUIV_MULTIPLIER=11.25,SEED=314,GPTQ_AR_CALIB_TEMP=0.9 slurm/train_march25_frontier_4gpu.sbatch
sbatch --export=ALL,RUN_ID=march25-proxy1125-s314-seqs96,H100_PROXY=1,H100_EQUIV_MULTIPLIER=11.25,SEED=314,GPTQ_AR_CALIB_SEQS=96 slurm/train_march25_frontier_4gpu.sbatch
```

Reason: the remaining matched-proxy gap is small enough that hardcoded export defaults are now an unnecessary bottleneck.

### Priority 3: avoid new architecture churn for the moment

I would not spend the next cycle on more architecture or optimizer changes yet. The stack is already close enough that we first need to answer:

- is late QAT helping at the matched budget?
- can better AR calibration recover most of the `~0.0017` gap?

If both answers are "not really," then it is worth revisiting training-side details. Until then, export is the cleaner bet.

## Decision Rule

- If tight-proxy no-QAT is clearly worse, keep late QAT in the parity stack.
- If no-QAT is neutral, treat export quality as the primary frontier bottleneck.
- If AR calibration sweeps recover most of the remaining gap, freeze the March 25 parity stack and move the int4 QAT work onto that exact base.
- If AR calibration sweeps stay flat, the next likely gains are hidden in training/runtime details rather than missing top-level features.

## Overnight Results (2026-04-03)

The overnight batch finished with five new logs:

- `slurm/output/pg-march25-frontier-4gpu-54846.out` — `seed=314`, no-QAT
- `slurm/output/pg-march25-frontier-4gpu-54847.out` — `seed=42`, no-QAT
- `slurm/output/pg-march25-frontier-4gpu-54848.out` — `seed=314`, `GPTQ_AR_CALIB_TEMP=0.7`
- `slurm/output/pg-march25-frontier-4gpu-54849.out` — `seed=314`, `GPTQ_AR_CALIB_SEQS=96`
- `slurm/output/pg-march25-frontier-4gpu-54850.out` — `seed=314`, `GPTQ_AR_CALIB_TEMP=0.9`

### Result Table

| Log | Change vs matched baseline | Steps | Float sliding | Int6 sliding | Size | Read |
|-----|----------------------------|-------|---------------|--------------|------|------|
| `54846` | no-QAT, seed `314` | `6924` | `1.11201207` | `1.11608931` | `15,989,478` | best clean run |
| `54847` | no-QAT, seed `42` | `7263` | `1.11164656` | `1.11553640` | `15,842,310` | faster-than-baseline, not apples-to-apples |
| `54848` | temp `0.7`, seed `314` | `6924` | `1.11234342` | `1.11682006` | `15,851,622` | worse |
| `54849` | seqs `96`, seed `314` | `6840` | `1.11421457` | `1.11780588` | `16,001,578` | clearly worse |
| `54850` | temp `0.9`, seed `314` | `6920` | `1.11224962` | `1.11611186` | `15,852,494` | promising |

### What Changed

#### 1. No-QAT looks real on seed `314`

Compare `54846` against the clean late-QAT seed-314 baseline `54843`:

- float sliding improved from `1.11234307` to `1.11201207` (`-0.00033100`)
- int6 sliding improved from `1.11648884` to `1.11608931` (`-0.00039953`)

This is the clearest clean signal from the overnight batch. The tradeoff is artifact size: `54846` grew to `15,989,478` bytes, about `150 KB` larger than `54843`.

#### 2. The seed-42 no-QAT run is directionally encouraging but not clean

`54847` finished much faster than the matched-proxy baseline:

- `7263` steps vs the earlier seed-42 matched baseline at `6930`
- average step time collapsed from a bad early start to `929.46 ms/step` by the end

It scored well, but it is not a fair matched-budget comparison because it got `333` extra steps. Treat it as a directional hint that no-QAT may help, not as the decisive seed-42 datapoint.

#### 3. Export sweeps: `temp=0.9` good, `temp=0.7` bad, `seqs=96` bad

Against the clean late-QAT seed-314 baseline `54843`:

- `temp=0.9` (`54850`) improved int6 sliding from `1.11648884` to `1.11611186` (`-0.00037698`)
- `temp=0.7` (`54848`) regressed to `1.11682006`
- `seqs=96` (`54849`) regressed badly to `1.11780588`

`temp=0.9` is especially interesting because it nearly matches the no-QAT gain while keeping the late-QAT training recipe and avoiding the larger no-QAT artifact.

### Updated Read

- best clean result so far: `54846` no-QAT on seed `314`
- best clean export-only tweak so far: `54850` with `GPTQ_AR_CALIB_TEMP=0.9`
- things to stop exploring in the near term: `GPTQ_AR_CALIB_TEMP=0.7` and `GPTQ_AR_CALIB_SEQS=96`

### Next Runs

The highest-value next run is now the obvious combination test:

```bash
sbatch --export=ALL,RUN_ID=march25-proxy1125-s314-noqat-temp09,H100_PROXY=1,H100_EQUIV_MULTIPLIER=11.25,SEED=314,LATE_QAT_THRESHOLD=0,GPTQ_AR_CALIB_TEMP=0.9 slurm/train_march25_frontier_4gpu.sbatch
```

After that:

```bash
sbatch --export=ALL,RUN_ID=march25-proxy1125-s314-temp095,H100_PROXY=1,H100_EQUIV_MULTIPLIER=11.25,SEED=314,GPTQ_AR_CALIB_TEMP=0.95 slurm/train_march25_frontier_4gpu.sbatch
sbatch --export=ALL,RUN_ID=march25-proxy1125-s42-noqat-rerun,H100_PROXY=1,H100_EQUIV_MULTIPLIER=11.25,SEED=42,LATE_QAT_THRESHOLD=0 slurm/train_march25_frontier_4gpu.sbatch
```

The rerun matters only if we want a clean seed-42 confirmation. If cluster time is tight, the combination run is more valuable than another broad export sweep.

## Follow-Up Notes (2026-04-03)

After reviewing the overnight logs again, the current read is a little sharper:

- the strongest clean levers are still `LATE_QAT_THRESHOLD=0` and `GPTQ_AR_CALIB_TEMP=0.9`
- this now looks more like an export-pressure problem than a missing-architecture problem
- the current `0.15` late-QAT onset may simply be too aggressive for the matched local budget

### Recommended next experiment

Run the obvious combination test first:

```bash
sbatch --export=ALL,RUN_ID=march25-proxy1125-s314-noqat-temp09,H100_PROXY=1,H100_EQUIV_MULTIPLIER=11.25,SEED=314,LATE_QAT_THRESHOLD=0,GPTQ_AR_CALIB_TEMP=0.9 slurm/train_march25_frontier_4gpu.sbatch
```

Why this one first:

- `54846` showed `no-QAT` was the best clean training-side result on seed `314`
- `54850` showed `temp=0.9` was the best clean export-only tweak on seed `314`
- combining the two best single-axis wins is higher value than widening the export sweep right now

### Narrow follow-ups if the combo is good but not enough

- test milder late-QAT instead of only `on` vs `off`: `LATE_QAT_THRESHOLD=0.05` and `0.10`
- test `GPTQ_AR_CALIB_TEMP=0.95` and `1.0`, then stop if flat
- try `GPTQ_AR_CALIB_SEED` decoupled from `SEED` to see whether calibration-sample variance matters

### Things that do not look worth prioritizing right now

- broader `GPTQ_AR_CALIB_SEQS` increases; `96` was clearly worse
- colder calibration temperatures like `0.7`
- new architecture churn before the export / late-QAT interaction is understood

## Combo Follow-Up Results (2026-04-03 evening)

Three more runs now sharpen the picture:

- `slurm/output/pg-march25-frontier-4gpu-54855.out` — `seed=314`, `LATE_QAT_THRESHOLD=0`, `GPTQ_AR_CALIB_TEMP=0.9`
- `slurm/output/pg-march25-frontier-4gpu-54856.out` — `seed=42`, `LATE_QAT_THRESHOLD=0`, `GPTQ_AR_CALIB_TEMP=0.9`
- `slurm/output/pg-march25-frontier-4gpu-54857.out` — `seed=42`, clean no-QAT rerun with default `temp=0.8`
- `slurm/output/pg-march25-frontier-4gpu-54861.out` — `seed=314`, `LATE_QAT_THRESHOLD=0.10`, `GPTQ_AR_CALIB_TEMP=0.9`

### Result read

#### 1. The combo run on seed `314` was not trustworthy as a clean quality datapoint

`54855` finished with only `6745` steps and regressed badly:

- float sliding `1.11428363`
- int6 sliding `1.11797646`

That is much worse than the earlier matched-budget seed-314 runs and is best treated as a slow-node / runtime-budget miss, not as strong evidence that the combo is intrinsically bad.

#### 2. The combo run on seed `42` looked clean but flat

`54856` landed at:

- `6927` steps
- float sliding `1.11251045`
- int6 sliding `1.11640480`

Compared with the earlier matched seed-42 baseline `54844`:

- baseline int6 sliding `1.11637954`
- combo int6 sliding `1.11640480`

That is effectively flat. On seed `42`, `no-QAT + temp=0.9` did not produce a meaningful gain.

#### 3. The unresolved question at that point was seed-42 no-QAT without the temp tweak

`54857` matters more than another temperature sweep because it isolates whether the seed-42 story is:

- no-QAT helps, but `temp=0.9` gives it back
- no-QAT is neutral on seed `42`
- or the earlier fast seed-42 no-QAT result was mostly runtime luck

### Updated decision rule at the time

- trust `54856` as the current clean combo verdict: not additive enough to prioritize
- treat `54855` as runtime-contaminated because the step budget fell well short
- wait for `54857` before deciding whether no-QAT should replace the late-QAT parity default

### Best next run after `54857` at that point

If `54857` confirms clean seed-42 no-QAT gains, the next best run is a milder late-QAT test rather than another export sweep:

```bash
sbatch --export=ALL,RUN_ID=march25-proxy1125-s314-qat010-temp09,H100_PROXY=1,H100_EQUIV_MULTIPLIER=11.25,SEED=314,LATE_QAT_THRESHOLD=0.10,GPTQ_AR_CALIB_TEMP=0.9 slurm/train_march25_frontier_4gpu.sbatch
```

Why:

- current evidence suggests the existing `0.15` late-QAT onset may be too aggressive
- the clean seed-314 wins came from either `no-QAT` or export tuning, not from stacking both
- `0.10` is a good test of whether "less QAT" beats both `off` and `0.15`

If `54857` comes back flat, freeze the export sweep for now and move to training-side micro-ablations like `LATE_QAT_THRESHOLD=0.10` or `0.05`.

## Final Read After `54857` And `54861` (2026-04-04)

The remaining ambiguity is now mostly gone.

### 1. Clean no-QAT won across both seeds

`54857` finished cleanly at:

- `6924` steps
- float sliding `1.11210100`
- int6 sliding `1.11617889`

Compared with the matched seed-42 late-QAT baseline `54844`:

- baseline int6 sliding `1.11637954`
- clean no-QAT int6 sliding `1.11617889`

That is a gain of `-0.00020065` BPB on seed `42`.

Together with seed `314` no-QAT run `54846` (`1.11608931`), the clean two-seed no-QAT mean is:

- no-QAT mean: `1.11613410`
- late-QAT matched baseline mean: `1.11643419`
- no-QAT advantage: `-0.00030009`

### 2. Milder late-QAT did not recover the loss

`54861` tested `LATE_QAT_THRESHOLD=0.10` with `GPTQ_AR_CALIB_TEMP=0.9` and came back clearly worse:

- `6808` steps
- float sliding `1.11408688`
- int6 sliding `1.11751938`
- size `16,048,254` bytes

This is worse on both quality and artifact size, so the `0.10` onset should not be pursued further.

### 3. Current working verdict

- promote `LATE_QAT_THRESHOLD=0` to the working matched-proxy baseline
- treat `GPTQ_AR_CALIB_TEMP=0.9` as a plausible export-only tweak, but not a proven additive gain on top of no-QAT
- stop spending time on legacy late-QAT onset sweeps for this stack

### Recommended next step

If we want one last export adjudication run before freezing the stack, rerun the seed-314 combo on a clean node:

```bash
sbatch --export=ALL,RUN_ID=march25-proxy1125-s314-noqat-temp09-rerun,H100_PROXY=1,H100_EQUIV_MULTIPLIER=11.25,SEED=314,LATE_QAT_THRESHOLD=0,GPTQ_AR_CALIB_TEMP=0.9 slurm/train_march25_frontier_4gpu.sbatch
```

Otherwise the cleaner move is to freeze the local parity base as `no-QAT` and move on to the int4-QAT port.

## Closeout After `54865` To `54868` (2026-04-04)

Four more logs closed the remaining March 25 questions:

- `slurm/output/pg-march25-frontier-4gpu-54865.out` — overlapping rerun of `seed=314`, `LATE_QAT_THRESHOLD=0`, `GPTQ_AR_CALIB_TEMP=0.9`
- `slurm/output/pg-march25-frontier-4gpu-54866.out` — second overlapping rerun of the same config
- `slurm/output/pg-march25-frontier-4gpu-54867.out` — `seed=314`, `LATE_QAT_THRESHOLD=0`, `GPTQ_AR_CALIB_TEMP=0.8`, `GPTQ_AR_CALIB_SEED=42`
- `slurm/output/pg-march25-frontier-4gpu-54868.out` — `seed=314`, `LATE_QAT_THRESHOLD=0`, `GPTQ_AR_CALIB_TEMP=0.9`, `GPTQ_AR_CALIB_SEED=42`

### Result table

| Log | Change | Steps | Float sliding | Int6 sliding | Size | Read |
|-----|--------|-------|---------------|--------------|------|------|
| `54865` | no-QAT + temp `0.9` rerun | `5953` | `1.11861368` | `1.12224883` | `15,880,338` | runtime-contaminated |
| `54866` | no-QAT + temp `0.9` rerun | `6932` | `1.11196138` | `1.11587877` | `15,856,990` | best clean seed-314 export result |
| `54867` | no-QAT + calib seed `42` | `6566` | `1.11550421` | `1.11925238` | `15,908,078` | runtime-contaminated and worse |
| `54868` | no-QAT + temp `0.9` + calib seed `42` | `6925` | `1.11237194` | `1.11628764` | `15,856,506` | clean but worse than calib seed `314` |

### Final read

#### 1. The clean `seed=314` rerun confirms the earlier no-QAT story

`54866` is the useful rerun, not `54865`.

Compared with the clean no-QAT seed-314 baseline `54846`:

- `54846`: float sliding `1.11201207`, int6 sliding `1.11608931`
- `54866`: float sliding `1.11196138`, int6 sliding `1.11587877`

That is a small but real-looking improvement for `GPTQ_AR_CALIB_TEMP=0.9` on seed `314`:

- float sliding: `-0.00005069`
- int6 sliding: `-0.00021054`

So `temp=0.9` remains plausible as a seed-314 export tweak, but the effect is modest.

#### 2. The overlapping rerun pattern made `54865` unusable

`54865` and `54866` shared the same `RUN_ID` and run directory and overlapped in time, just like the earlier `54854` and `54855` pair. `54865` only reached `5953` steps and should be treated as a bad runtime datapoint, not as evidence about the recipe.

Practical lesson: do not submit concurrent jobs with the same `RUN_ID` again when collecting ablations.

#### 3. Calibration-seed sweeps do not look worth more time

`54868` is the only clean calibration-seed test in this mini-batch, and it lost to `54866`:

- `54866` (`temp=0.9`, calib seed `314`): int6 sliding `1.11587877`
- `54868` (`temp=0.9`, calib seed `42`): int6 sliding `1.11628764`

That is a regression of `+0.00040887` BPB.

`54867` also looked bad, but it only reached `6566` steps, so the clean conclusion should come from `54868`: decoupling `GPTQ_AR_CALIB_SEED` from the training seed is not a promising direction on this stack.

#### 4. The March 25 verdict is now stable enough to freeze

The useful clean signals across the full sweep are:

- no-QAT beats late-QAT at matched proxy
- `GPTQ_AR_CALIB_TEMP=0.9` is at best a small export-only gain, not a robust new default
- broader export sweeps like `seqs=96` and calibration-seed changes are not paying off
- the main residual variance is throughput / step-count noise, not missing feature flags

### Updated recommendation

Freeze the local matched-proxy parity base as:

- `H100_EQUIV_MULTIPLIER=11.25`
- `LATE_QAT_THRESHOLD=0`
- `GPTQ_AR_CALIB_TEMP=0.8` as the conservative default

Treat this as the optional export-only sidecar:

- `GPTQ_AR_CALIB_TEMP=0.9` on the same frozen no-QAT stack when checking whether a specific seed benefits

Do not spend more time on:

- legacy late-QAT onset sweeps
- `GPTQ_AR_CALIB_SEQS > 64`
- calibration-seed sweeps
- concurrent reruns that share a `RUN_ID`

## Recommended Next Experiments

### 1. Finish the local 3-seed no-QAT matched-proxy baseline

The record comparison is still 3-seed, while the clean local no-QAT read is only 2-seed.

```bash
sbatch --export=ALL,RUN_ID=march25-proxy1125-s999-noqat,H100_PROXY=1,H100_EQUIV_MULTIPLIER=11.25,SEED=999,LATE_QAT_THRESHOLD=0 slurm/train_march25_frontier_4gpu.sbatch
```

Why first:

- it closes the parity bookkeeping cleanly
- it tells us whether the current local gap is still real at the 3-seed level
- it gives the int4 port a better control than a 2-seed mean

### 2. Port int4 Hadamard / trust-gradient QAT onto the frozen no-QAT base

Once the 3-seed no-QAT control is in place, move the new idea onto the exact stack that won this sweep instead of continuing legacy late-QAT work.

Start with a narrow seed-314 matrix:

- `QAT=off` control on the frozen no-QAT base
- int4 QAT with onset `0.15`
- int4 QAT with onset `0.20`

Why this matrix:

- `0.10` already looked too aggressive in the int6-style late-QAT setting
- `0.15` is the historical frontier default
- `0.20` tests whether the new int4 path wants an even later onset on the tighter local budget

### 3. Only keep one small export sidecar alive

If we want one non-int4 sidecar while the port lands, make it:

```bash
sbatch --export=ALL,RUN_ID=march25-proxy1125-s999-noqat-temp09,H100_PROXY=1,H100_EQUIV_MULTIPLIER=11.25,SEED=999,LATE_QAT_THRESHOLD=0,GPTQ_AR_CALIB_TEMP=0.9 slurm/train_march25_frontier_4gpu.sbatch
```

This is only worth doing if the `seed=999` no-QAT control looks strong enough that a `temp=0.9` confirmatory run could realistically become the frozen export default.

### 4. Stop broad March 25 churn after that

If the 3-seed no-QAT mean still sits around the current local plateau, the right move is not more March 25 micro-sweeps. The right move is to use this stack as the stable base for:

- the int4 QAT ablation
- later artifact-headroom or scale-up work if int4 earns it
