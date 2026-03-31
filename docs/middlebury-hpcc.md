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

## Train on 4 GPUs

The main training script supports distributed launch with `torchrun`. On Ada, the natural scale-up target is one full 4-GPU node:

```bash
sbatch slurm/train_4gpu.sbatch
```

Example with overrides:

```bash
sbatch --export=ALL,RUN_ID=baseline-a6000 slurm/train_4gpu.sbatch
```

Important note: the challenge leaderboard target is 8xH100 in under 10 minutes. Ada's current GPU nodes are 4x RTX A6000 or 4x RTX A5000, so cluster runs are great for experimentation and scaling studies, but not a hardware match for the official benchmark.

## Monitoring jobs

Useful commands:

```bash
squeue -u "$USER"
sacct -j <jobid> --format=JobID,JobName,Partition,Elapsed,State,ExitCode
srun --overlap --jobid=<jobid> --pty nvidia-smi
```

## Outputs

The SLURM scripts in [slurm](/home/pkcutler/parameter-golf/slurm) write each run into:

```bash
$STORAGE/parameter-golf-runs/<run-id>
```

That directory will contain:

- `slurm-<jobid>.out`
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
