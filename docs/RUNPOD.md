# Runpod / remote GPU workflow

This is the clean path for serious Parameter Golf work.

Local WSL is fine for repo setup, note-taking, and maybe tiny smoke debugging, but serious experiments should run on a remote CUDA box.

## Goal

Use a remote GPU machine to:
- reproduce the baseline cleanly
- run structured ablations
- save logs + metrics in a repeatable way

## Recommended split

- `~/Projects/parameter-golf` = training code and record folders
- `~/Projects/autoresearch` = ideas, experiment planning, result synthesis

## 1. Create a remote box

The upstream repo recommends Runpod. Good first target:
- **1x H100** for sanity / ablations

For leaderboard-style runs later:
- **8x H100**

Use the official template from the repo README if available.

## 2. SSH into the remote machine

You should land in `/workspace` or another local-disk working directory.

## 3. Clone the repo on the remote box

```bash
cd /workspace
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
```

If you want your local patches/work instead of pristine upstream:
- either push your branch somewhere first, or
- copy your modified files over with `scp`/`rsync`

## 4. Download the cached dataset

Small smoke subset:
```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
```

Full baseline-style dataset:
```bash
python3 data/cached_challenge_fineweb.py --variant sp1024
```

## 5. First 1-GPU baseline sanity run

```bash
RUN_ID=baseline_sp1024_1gpu \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## 6. 8-GPU leaderboard-style run

```bash
NCCL_IB_DISABLE=1 \
RUN_ID=baseline_sp1024_8gpu \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## 7. Pull logs/results back

After a run, copy back at least:
- `logs/<run>.txt`
- `final_model.int8.ptz` if useful
- any record folder / notes

Example:
```bash
scp root@YOUR_REMOTE_HOST:/workspace/parameter-golf/logs/baseline_sp1024_1gpu.txt ./remote-logs/
```

## 8. Parse a run log locally

Back on your laptop:

```bash
cd ~/Projects/parameter-golf
source .venv/bin/activate
python3 scripts/parse_train_log.py logs/<run>.txt
```

## Suggested experiment ladder

1. **Remote smoke**
   - 1 shard
   - 1 GPU
   - confirm end-to-end training works
2. **Remote baseline sanity**
   - 1 GPU
   - default architecture
3. **Cheap ablations**
   - width/depth/head/GQA changes
   - LR schedule tweaks
4. **Longer non-record runs**
   - identify promising configs
5. **8-GPU record-style runs**
   - only for best candidates

## What to avoid

- don’t treat local WSL timing as meaningful for leaderboard work
- don’t start with 8x H100 before the 1x H100 path is clean
- don’t fork the training code into `autoresearch`; keep it adjacent

## Practical next step

The next real milestone is:
- get one clean remote 1-GPU run
- parse the resulting log
- record the result in the experiment tracker
