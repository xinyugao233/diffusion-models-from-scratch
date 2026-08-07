# Epsilon-Prediction Training and Tiny-Subset Overfit Plan

## Status and authority

This is the authoritative implementation and execution plan for Milestone 3
and `EXP001/try01`. It was frozen before training code was implemented or
either optimization gate was run. The validated U-Net architecture and DDPM
forward equations remain unchanged.

## Research question

Can the existing smoke U-Net optimize the DDPM epsilon-prediction objective,
first on one fixed synthetic noising problem and then on a fixed set of 16
normalized CIFAR-10 training images?

## Motivation and existing evidence

Milestone 1 validated the linear schedule, coefficient extraction, `q_sample`,
and exact-noise reconstruction. Milestone 2 validated the time-conditioned
U-Net's shapes, gradients, parameter budget, and CPU execution. Neither
milestone performed an optimizer update, so training wiring and small-dataset
learnability remain untested.

## Hypothesis and interpretation rule

If each noised input is constructed with a sampled noise tensor and that exact
tensor is also used as the MSE target, AdamW updates to the smoke U-Net will
produce finite gradients, change parameters, and reduce loss. Passing both
gates supports only this conclusion:

> The complete CIFAR-10 epsilon-prediction pipeline can learn a fixed small
> dataset.

It does not establish reverse-process correctness, sample quality,
generalization, full-dataset convergence, or useful FID.

## Baseline, intervention, and controls

- Baseline: commit `26e735e597001251d92075d5bf02fb5649851800` with no
  optimization code or result.
- Intervention: explicit epsilon-prediction MSE and AdamW parameter updates.
- Control A: a fully fixed synthetic batch, timesteps, and noise, with `x_t`
  constructed once.
- Control B: exactly 16 fixed CIFAR-10 training examples with no augmentation;
  only timesteps and noise vary between steps.
- No comparison against another schedule, architecture, target, or optimizer
  is part of this try.

## Frozen scientific invariants

- Image tensors have shape `[B,3,32,32]` and model-space range `[-1,1]`.
- The schedule is linear with `T=1000`, `beta_start=1e-4`, and
  `beta_end=2e-2`; code timestep 0 applies the first beta.
- The model is the unchanged 491,107-parameter smoke U-Net from Milestone 2.
- One independently sampled timestep is used per image.
- The exact same noise tensor constructs `x_t` and supplies the MSE target.
- Gate A constructs `x_t` once and reuses it.
- Gate B uses fresh timesteps and fresh Gaussian noise at every step.
- Gate B uses no random horizontal flip or other augmentation.
- Runs use CPU only. No external job, GPU, full-dataset training, EMA,
  checkpointing, DDPM reverse sampling, DDIM, AMP, or distributed training is
  permitted.

## Gate A: deterministic synthetic optimization

Frozen configuration:

| Field | Value |
|---|---|
| Model seed | `7` |
| Synthetic-data/noise seed | `11` |
| Batch size | `4` |
| Fixed timesteps | `[0, 100, 500, 999]` |
| Optimizer | AdamW, betas `(0.9, 0.999)`, eps `1e-8`, weight decay `0` |
| Learning rate | `1e-3` |
| Steps | `40` |
| Gradient clip | global L2 norm `1.0` |
| Early/late windows | first/last `5` losses |
| Device | CPU |

Exact command:

```bash
.venv/bin/python scripts/run_synthetic_optimization.py \
  --config configs/synthetic_optimization.json
```

Acceptance requires all losses and gradient norms to be finite, at least one
parameter to change, and late-window mean loss to be at most `0.80` times the
early-window mean. The 20% threshold is large relative to step-to-step numeric
noise because the entire training example is fixed. The normal test suite must
also pass before Gate B.

The smoke U-Net contains no attention. A separate two-update diagnostic on the
existing zero-initialized `SelfAttention2d` records whether its initially zero
upstream QKV gradient becomes nonzero after the output projection updates; this
does not alter either model configuration.

## Gate B: fixed 16-image CIFAR-10 overfit

Frozen configuration:

| Field | Value |
|---|---|
| Dataset | CIFAR-10 training split |
| Dataset root | `data` |
| Subset indices | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]` |
| Labels | Filled into the manifest from verified dataset metadata before the run |
| Model/training seed | `23` |
| Per-step sampling seed | `24` |
| Batch size | `16` |
| Optimizer | AdamW, betas `(0.9, 0.999)`, eps `1e-8`, weight decay `0` |
| Learning rate | `1e-3` |
| Steps | `500` |
| Gradient clip | global L2 norm `1.0` |
| Early/late windows | first/last `50` losses |
| Device | CPU |
| Baseline Git commit | `26e735e597001251d92075d5bf02fb5649851800` |

Exact command:

```bash
.venv/bin/python scripts/run_overfit_16.py \
  --config configs/overfit_16.json \
  --download
```

Acceptance requires a normal exit, no nonfinite loss or gradient, and
late-window mean loss at most `0.80` times the early-window mean. A 20% windowed
reduction is frozen as a substantial signal beyond individual stochastic
minibatch fluctuations while remaining appropriate for only 500 CPU updates.
Failure stops the milestone; changing this threshold or configuration requires
a new numbered try rather than reinterpretation after seeing results.

## Dataset and manifest procedure

Gate B begins only after Gate A and all fast tests pass. The loader is invoked
with download enabled. The 16 indices above are immutable for this try. Their
integer labels and class names are read from the training dataset, verified to
match the requested count, and saved in
`configs/cifar10_overfit_16_manifest.json` before optimization begins.

## Implementation steps

1. Add a small training module that builds explicit epsilon targets, computes
   scalar MSE, rejects nonfinite values, calculates global gradient norm,
   optionally clips gradients, performs one optimizer step, and logs scalars.
2. Add synthetic-only unit tests for normalization, deterministic supplied
   noise, shapes, parameter updates, input validation, nonfinite detection, and
   clipping.
3. Add frozen JSON configurations and the two gate scripts.
4. Run the full fast test/lint/format gate and Gate A.
5. If Gate A passes, prepare the manifest and run Gate B only.
6. Preserve raw JSONL/CSV logs, summary JSON, a PNG loss curve, exact source
   hashes, and environment/device details under `EXP001/try01`.
7. Inspect the curve and logs, then update the try report, metadata, milestone
   report, project state, experiment index, and timeline.

## Expected artifacts and counts

- Gate A: 40 scalar records and one summary JSON.
- Attention diagnostic: two gradient-state records within the Gate A summary.
- Gate B: 500 scalar records, one summary JSON, one loss CSV, and one loss-curve
  PNG.
- Manifest: exactly 16 unique indices and 16 corresponding labels.
- All outputs belong to `Experiments/exp001-epsilon-prediction-learnability/try01/`.

## Failure and stop conditions

- Stop before Gate B if tests, lint, formatting, Gate A finiteness, parameter
  change, or the Gate A windowed-loss criterion fails.
- Stop either run immediately on a nonfinite loss or gradient.
- Stop if the dataset manifest differs from 16 unique fixed indices or if
  images are outside the expected normalized range.
- Do not relax acceptance criteria, add training steps, or switch device after
  seeing results in this try.
- Stop after Milestone 3; do not begin later roadmap milestones.

## Acceptance criteria

1. All requested unit tests pass without dataset access.
2. Ruff lint, Ruff format check, and `git diff --check` pass.
3. Gate A satisfies its frozen windowed-loss and parameter-change criteria.
4. Gate B satisfies its frozen windowed-loss and completion criteria.
5. Logs contain the expected record counts and finite values.
6. The fixed manifest and all configuration/provenance fields are durable.
7. Documentation states the exact supported and unsupported conclusions.

## Compute and documentation plan

Both authorized runs are local CPU jobs. The run will record baseline commit,
dirty working-tree status, SHA-256 hashes of executed code/configuration,
PyTorch/Python versions, device, duration, and exact commands because the user
did not authorize an implementation commit before execution. The final report
will explicitly distinguish the committed baseline from the uncommitted code
identity used for the run.
