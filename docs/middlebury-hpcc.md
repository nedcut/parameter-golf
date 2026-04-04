# Running Parameter Golf on Middlebury HPCC

This repo already includes the CUDA training path in [train_gpt.py](/home/pkcutler/parameter-golf/train_gpt.py), so the main cluster work is:

1. Put the dataset somewhere shared by compute nodes.
2. Activate a Python environment with CUDA-enabled PyTorch.
3. Submit GPU jobs through SLURM.

This guide is tailored to Ada as observed on 2026-03-30:

- login node: `ada`
- GPU partitions: `gpu-short`, `gpu-standard`, `gpu-long`
- GPU nodes currently expose either `4x RTX A6000` or `4x RTX A5000`

The official Middlebury docs for general SLURM and GPU usage are:

- https://sites.middlebury.edu/hpcc/documentation/
- https://sites.middlebury.edu/hpcc/getting-started/

## Recommended Path In This Fork

If you only want the shortest route to the current competitive workflow in this fork, use this order:

1. Read [PLAN.md](/home/pkcutler/parameter-golf/PLAN.md) for the current frozen baseline and next experiments.
2. Read [notes/2026-04-02-march25-frontier-proxy.md](/home/pkcutler/parameter-golf/notes/2026-04-02-march25-frontier-proxy.md) for the March 25 local replication history and what has already been ruled out.
3. Use [slurm/train_march25_frontier_4gpu.sbatch](/home/pkcutler/parameter-golf/slurm/train_march25_frontier_4gpu.sbatch) for March 25 parity work.
4. Use [records/track_10min_16mb/2026-03-26_11L_PreTTT_Frontier_Int4QAT/README.md](/home/pkcutler/parameter-golf/records/track_10min_16mb/2026-03-26_11L_PreTTT_Frontier_Int4QAT/README.md) only when you are actively working on the newer int4-QAT scaffold.

As of 2026-04-04, the local March 25 working baseline is:

- `H100_PROXY=1`
- `H100_EQUIV_MULTIPLIER=11.25`
- `LATE_QAT_THRESHOLD=0`
- `GPTQ_AR_CALIB_TEMP=0.8` as the conservative export default
- `GPTQ_AR_CALIB_TEMP=0.9` as an optional sidecar tweak, not a promoted default

One operational rule that deserves to be explicit: always use a fresh `RUN_ID` for each submitted job. Reusing a `RUN_ID` causes multiple jobs to share the same run directory and log path, which makes ablations hard to trust.

## Storage layout

Use persistent cluster storage for the main dataset and run artifacts:

```bash
mkdir -p "$STORAGE/parameter-golf-data"
mkdir -p "$STORAGE/parameter-golf-runs"
```

Recommended convention:

- dataset root: `$STORAGE/parameter-golf-data`
- run outputs: `$STORAGE/parameter-golf-runs`

The job scripts fall back to `$SCRATCH` only if `$STORAGE` is unavailable.

The provided download job symlinks the repo's ignored `data/datasets` and `data/tokenizers` paths into that storage-backed location so the existing code continues to work unchanged.

## Python environment

This checkout currently uses a repo-local `.conda/` environment.
The provided SLURM scripts prefer it because it includes Python headers needed by Triton / `torch.compile` on the GPU nodes.

If you need a fresh env, Middlebury recommends conda for Python projects and shows GPU PyTorch setup in their docs. A typical setup is:

```bash
module load cuda/12.6
conda create --name parameter-golf python=3.11
conda activate parameter-golf
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you hit PyTorch install issues, install CUDA-enabled PyTorch first, then install the remaining requirements.

## Download the published FineWeb cache

The training script expects the dataset shards and tokenizer locally accessible on the cluster filesystem.

The included job script downloads the published cache into storage:

```bash
sbatch slurm/download_data_short.sbatch
```

By default it downloads the `sp1024` tokenizer family with `10` training shards, which is a good first iteration target. Override shard count at submission time if needed:

```bash
sbatch --export=ALL,TRAIN_SHARDS=80 slurm/download_data_short.sbatch
```

## Smoke test on 1 GPU

Run a quick validation that your environment, data path, and CUDA stack all work:

```bash
sbatch slurm/train_smoke_1gpu.sbatch
```

Useful submission-time overrides:

```bash
sbatch --export=ALL,DATA_ROOT=$STORAGE/parameter-golf-data,RUN_ID=my-smoke slurm/train_smoke_1gpu.sbatch
```

## Frontier scaffold smoke on 1 GPU

Use the March 26 pre-TTT frontier scaffold through its dedicated job:

```bash
sbatch slurm/train_frontier_smoke_1gpu.sbatch
```

The frontier smoke defaults to the explicit `QAT=off` control. Useful overrides:

It also defaults `EVAL_STRIDE=0` so the smoke job skips the expensive final sliding-window evaluation and finishes comfortably inside `gpu-short`.
The smoke script now also defaults `WARMDOWN_ITERS=$ITERATIONS`, so "late-onset" QAT tests are actually late in a 200-step smoke rather than activating immediately because of the full-run `3500`-step warmdown.
By default it enables a pre-EMA export diagnostic and writes extra `pre_ema_*` lines to the log.

```bash
sbatch --export=ALL,RUN_ID=frontier-smoke-noqat slurm/train_frontier_smoke_1gpu.sbatch
```

```bash
sbatch --export=ALL,RUN_ID=frontier-smoke-int4,QAT_BITS=4,QAT_ONSET_SCALE=0.15,QAT_BLOCK_SIZE=128 slurm/train_frontier_smoke_1gpu.sbatch
```

To launch a comparable smoke matrix across seeds and modes:

```bash
./scripts/submit_frontier_matrix.sh
```

Useful overrides:

```bash
TARGET=smoke MATRIX=full RUN_GROUP=frontier-smoke-a ./scripts/submit_frontier_matrix.sh
TARGET=smoke MATRIX=onset SEEDS="1337" ./scripts/submit_frontier_matrix.sh
```

To summarize the resulting logs:

```bash
python3 scripts/summarize_frontier_logs.py "slurm/output/pg-frontier-smoke-*.out"
```

## Train on 4 GPUs

The main training script supports distributed launch with `torchrun`. On Ada, the natural scale-up target is one full 4-GPU node:

```bash
sbatch slurm/train_4gpu.sbatch
```

Example with overrides:

```bash
sbatch --export=ALL,RUN_ID=baseline-a6000 slurm/train_4gpu.sbatch
```

## Train the frontier scaffold on 4 GPUs

Use the dedicated frontier job when you want the March 26 scaffold instead of the top-level trainer:

```bash
sbatch slurm/train_frontier_4gpu.sbatch
```

This job defaults to the no-QAT control and disables the 10-minute wallclock cap so the frontier stack can run to completion on Ada. Common overrides:
Unlike the smoke job, it keeps the full-run `WARMDOWN_ITERS=3500` default unless you override it.

If you want a rough local proxy for the official `10 minutes on 8xH100` budget, the 4-GPU frontier job now supports `H100_PROXY=1`.
By default this multiplies `600s` by `H100_EQUIV_MULTIPLIER=11.72`, which is based on the observed April 1 Ada run speed of about `1015 ms/step` versus the March 25 reference speed of about `86.6 ms/step`.
That yields `MAX_WALLCLOCK_SECONDS≈7032`, or about `117 minutes`.

```bash
sbatch --export=ALL,RUN_ID=frontier4-noqat slurm/train_frontier_4gpu.sbatch
```

```bash
sbatch --export=ALL,RUN_ID=frontier4-legacy-int6,LATE_QAT_THRESHOLD=0.15 slurm/train_frontier_4gpu.sbatch
```

```bash
sbatch --export=ALL,RUN_ID=frontier4-int4,QAT_BITS=4,QAT_ONSET_SCALE=0.15,QAT_BLOCK_SIZE=128 slurm/train_frontier_4gpu.sbatch
```

To run in H100-proxy mode:

```bash
sbatch --export=ALL,RUN_ID=frontier4-proxy,H100_PROXY=1 slurm/train_frontier_4gpu.sbatch
```

To override the slowdown estimate without changing the script:

```bash
sbatch --export=ALL,RUN_ID=frontier4-proxy-tuned,H100_PROXY=1,H100_EQUIV_MULTIPLIER=10.8 slurm/train_frontier_4gpu.sbatch
```

If you set `MAX_WALLCLOCK_SECONDS` explicitly, that still wins over `H100_PROXY=1`.

To submit the same matrix structure on 4 GPUs later:

```bash
TARGET=full MATRIX=baseline RUN_GROUP=frontier4-a ./scripts/submit_frontier_matrix.sh
```

## Run the March 25 frontier record stack on 4 GPUs

When you want the latest upstream legal frontier recipe rather than the older March 26 scaffold, use:

```bash
sbatch slurm/train_march25_frontier_4gpu.sbatch
```

This launcher defaults to the March 25 record settings that matter most on the local cluster:

- `BIGRAM_VOCAB_SIZE=3072`
- `BIGRAM_DIM=112`
- `WARMDOWN_ITERS=4000`
- `TARGET_MB=15.9`
- `SEED=42`

To run the best local proxy for the official 10-minute budget:

```bash
sbatch --export=ALL,RUN_ID=march25-proxy,H100_PROXY=1 slurm/train_march25_frontier_4gpu.sbatch
```

That keeps the March 25 model stack but replaces the default 600-second cap with the local H100-equivalent proxy budget from `H100_EQUIV_MULTIPLIER`.
For the March 25 record stack specifically, the current matched-budget local baseline is:

```bash
sbatch --export=ALL,RUN_ID=march25-proxy1125,H100_PROXY=1,H100_EQUIV_MULTIPLIER=11.25 slurm/train_march25_frontier_4gpu.sbatch
```

The launcher default is still `11.72`, but the tighter `11.25` setting matched the original March 25 step budget much more closely in the April 2-4 local replications.

For the current clean no-QAT baseline in this fork, use:

```bash
sbatch --export=ALL,RUN_ID=march25-proxy1125-s999-noqat,H100_PROXY=1,H100_EQUIV_MULTIPLIER=11.25,SEED=999,LATE_QAT_THRESHOLD=0 slurm/train_march25_frontier_4gpu.sbatch
```

If you want the only export-side follow-up that still looks worth checking, use:

```bash
sbatch --export=ALL,RUN_ID=march25-proxy1125-s999-noqat-temp09,H100_PROXY=1,H100_EQUIV_MULTIPLIER=11.25,SEED=999,LATE_QAT_THRESHOLD=0,GPTQ_AR_CALIB_TEMP=0.9 slurm/train_march25_frontier_4gpu.sbatch
```

At this point, do not prioritize:

- legacy late-QAT onset sweeps on the March 25 stack
- `GPTQ_AR_CALIB_SEQS > 64`
- calibration-seed sweeps

Those all looked flat or worse in the April 2-4 local ablations.

Important note: the challenge leaderboard target is 8xH100 in under 10 minutes. Ada's current GPU nodes are 4x RTX A6000 or 4x RTX A5000, so cluster runs are great for experimentation and scaling studies, but not a hardware match for the official benchmark.

## Monitoring jobs

Useful commands:

```bash
squeue -u "$USER"
sacct -j <jobid> --format=JobID,JobName,Partition,Elapsed,State,ExitCode
srun --overlap --jobid=<jobid> --pty nvidia-smi
```

## Outputs

The SLURM scripts in [slurm](/home/pkcutler/parameter-golf/slurm) keep the scheduler stdout/stderr copy under:

```bash
slurm/output/%x-%j.out
```

The actual training artifacts for each run still go into:

```bash
$STORAGE/parameter-golf-runs/<run-id>
```

That run directory will contain the trainer outputs, for example:

- `logs/<run-id>.txt`
- `final_model.pt`
- `final_model.int8.ptz`

## Picking partitions

- `gpu-short`: fastest queue to test with, 2 hour limit
- `gpu-standard`: better default for longer experiments, 2 day limit
- `gpu-long`: use when you explicitly need multi-day runs

## Notes about this repo

- [train_gpt.py](/home/pkcutler/parameter-golf/train_gpt.py) requires CUDA and will fail on the login node by design.
- The script expects `WORLD_SIZE` to divide 8, so `1`, `2`, or `4` GPUs work cleanly on this cluster.
- The repo currently does not have dataset shards checked into `data/datasets`, so plan on using the storage-backed paths from the SLURM scripts.
