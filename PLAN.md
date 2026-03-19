# Parameter Golf Plan

Goal: beat the current baseline (`1.2244 val_bpb`) reliably, then push toward a submission-worthy result without getting lost in premature weirdness.

## Current state

- Repo is at `~/Projects/parameter-golf`
- Local env works in `.venv`
- `run.sh` exists for common commands
- Baseline to beat: `1.2244 val_bpb`
- Existing unlimited-compute reference: `1.2074 val_bpb`

## Strategy

We should treat this like a tight engineering loop, not a grand-theory project.

1. **Reproduce and understand the baseline**
2. **Build a local experiment harness**
3. **Do cheap ablations first**
4. **Escalate only the promising ideas to bigger runs**
5. **Use `autoresearch` as the idea-tracker / experiment-planner, not as the core training repo**

## Phase 1 — Baseline reproduction

### Objective
Get one clean local/remote baseline run and extract the real optimization levers.

### Tasks
- [ ] Confirm dataset download completed and paths are valid
- [ ] Run a small local smoke test to verify the full loop
- [ ] Read `train_gpt.py` closely enough to identify:
  - model shape knobs
  - optimizer/schedule knobs
  - quantization/compression path
  - logging + validation path
  - wallclock stop logic
- [ ] Run one baseline-like single-GPU experiment for sanity
- [ ] If we use Runpod or another remote box, run one faithful baseline reproduction there

### Deliverables
- `notes/baseline.md` or equivalent run notes
- A table of tunable variables and their expected effect on:
  - quality
  - speed
  - compressed size

## Phase 2 — Measurement harness

### Objective
Make experiments cheap to compare.

### Tasks
- [ ] Create an experiment log format (CSV/JSONL/markdown)
- [ ] Track at minimum:
  - run id
  - code diff / commit
  - tokenizer
  - model size / architecture
  - training tokens seen
  - wallclock
  - final `val_loss`
  - final `val_bpb`
  - compressed model bytes
  - total artifact bytes
- [ ] Add a helper script to snapshot key metrics from stdout/train logs
- [ ] Separate:
  - smoke experiments
  - serious local runs
  - remote leaderboard-style runs

### Deliverables
- `experiments/` or `notes/` folder with structured logs
- lightweight parser/helper for result extraction

## Phase 3 — High-probability improvements first

These are the first ideas worth testing because they are less deranged than inventing an entirely new architecture on day one.

### 3.1 Architecture-size frontier search
Goal: find a better point on the quality/compression tradeoff.

Try controlled sweeps over:
- `NUM_LAYERS`
- `MODEL_DIM`
- `NUM_HEADS`
- `NUM_KV_HEADS`
- `MLP_MULT`
- tied vs untied embeddings
- sequence length if allowed by throughput

Hypothesis:
- The current 9x512 baseline is probably not Pareto-optimal for compressed artifact size.
- Small changes in width/depth/head structure may improve bpb at the same compressed size.

### 3.2 Training schedule / optimizer tuning
Try:
- LR and warmup
- tied embedding LR
- weight decay
- batch tokens
- validation cadence (for instrumentation, not necessarily for final runs)
- longer runs in non-record mode to identify better configs, then distill back to 10-minute runs

Hypothesis:
- There is probably free performance in schedule tuning before architecture novelty is required.

### 3.3 Quantization/compression-aware tuning
Inspect exactly how `final_int8_zlib_roundtrip` is computed.

Try:
- parameter distributions that compress better
- regularization that encourages compressibility
- low-rank / tied / shared weights
- architectural choices that reduce entropy in final weights

Hypothesis:
- Since the target metric includes compressed artifact size, boring ML improvements alone may leave score on the table.

## Phase 4 — More aggressive ideas

Only do these after we have a decent harness and a few wins.

### Candidate ideas
- recurrent depth / repeated blocks with shared weights
- heavier parameter tying
- factorized embeddings or low-rank layers
- tiny latent state + more test-time compute
- bitnet / constrained-weight experiments
- alternate tokenizer experiments if the repo rules + measurement story are clean enough

These are attractive, but they can easily become a swamp.

## Collaboration model with `~/Projects/autoresearch`

Use `autoresearch` as the adjacent research notebook/orchestrator repo.

### Good use of `autoresearch`
- tracking ideas
- ranking experiments
- maintaining run summaries
- synthesizing observations from logs
- proposing next ablations based on results

### Bad use of `autoresearch`
- replacing the actual `parameter-golf` training code too early
- creating a giant meta-framework before we have baseline competence

### Recommended workflow
1. Run training/code changes in `~/Projects/parameter-golf`
2. Save results/log summaries
3. Use `~/Projects/autoresearch` to:
   - analyze results
   - propose next experiments
   - keep a living research agenda

## Near-term next steps

1. Finish/verify the smoke dataset download
2. Run a local smoke test end-to-end
3. Read + annotate `train_gpt.py`
4. Set up result logging
5. Do the first 5–10 cheap ablations
6. Promote the best 1–2 configs to bigger runs

## Success criteria

### Short-term
- Reproduce baseline behavior cleanly
- Beat `1.2244` locally or in non-record runs

### Medium-term
- Find a configuration consistently near or below `1.21`
- Understand which gains come from:
  - architecture
  - schedule
  - compressibility tricks

### Stretch
- Produce a clean record-style folder under `records/` with reproducible logs and submission metadata

## My opinionated take

The fastest path to something good is probably:
- mild architecture sweeps
- more careful optimization tuning
- targeted compressibility-aware tweaks

Not “invent a moonshot architecture immediately.” The weird stuff can come after we’ve earned the right to be weird.
