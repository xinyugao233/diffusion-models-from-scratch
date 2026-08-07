# EXP004 Hypothesis: Full CIFAR-10 DDPM Training

## Primary hypothesis

The unchanged validated DDPM components, when orchestrated in a deterministic
production trainer, will train stably on all CIFAR-10 training images and yield
EMA samples with increasingly coherent CIFAR-like structure across the fixed
10k, 25k, and 50k checkpoint progression.

## Motivation

Component tests cannot establish that real-data ordering, long optimization,
resume, EMA checkpoint evaluation, and the full sampler work together.

## Predicted result

- Gate A and Gate B complete with finite losses and gradients.
- Gate B reconstructs and resumes in a new Slurm job and produces exactly
  reproducible fixed-seed samples.
- The final 1,000-step mean loss is below 90% of the first 1,000-step mean.
- Fixed-seed EMA grids become visibly less noise-like and more object-structured
  from 10k through 50k.

## Alternative explanations

- Loss may decrease while sampling is wrong because training and sampling
  parameterizations disagree; the already validated sampler and fixed-seed
  checkpoint evaluations control this risk.
- Loss may decrease without recognizable generation because the baseline is
  undertrained or its schedule/model is insufficient.
- Apparent progression may be seed selection; all checkpoint grids reuse the
  same preregistered initial tensors and reverse seed.
- Resume may execute but change data ordering; batching is derived from global
  step and a fixed epoch seed.

## Required controls

- Exact source/config/dataset identity.
- Finite loss/gradient/parameter/EMA checks.
- Two-process Gate B resume with configuration guard.
- Same fixed initial tensors and reverse seed at every evaluation.
- Full raw loss history and unchanged checkpoint artifacts.

## Falsification conditions

Nonfinite optimization state, failed resume, source/config drift, dataset
identity mismatch, failure of the frozen loss-reduction gate, or noise-like
50k EMA samples falsifies the corresponding stability or full-system claim.

## Interpretation rules

Stable optimization and recognizable samples support only an end-to-end
baseline claim. No FID, comparative, state-of-the-art, DDIM, or generalization
claim is permitted.

## Stop conditions

Follow `docs/plans/full_cifar10_training.md`. Poor early samples alone do not
stop training; structural invalidity does. Do not exceed 50k automatically.

## Preregistration status

Frozen on 2026-08-07 before production-trainer implementation or execution.
