# EXP004 Full CIFAR-10 DDPM Training

## Status

`READY` — configuration, production trainer, Slurm entrypoints, and local tests
pass; no Slurm job has been submitted.

## Research question

Can the complete validated DDPM train stably on all CIFAR-10 training images
and produce increasingly structured or recognizable EMA samples?

## Motivation and program connection

Milestones 1–5 validate the components independently. This experiment is the
first full-data, end-to-end training and sampling test, using unchanged
mathematics and architecture.

## Design

- Baseline: Milestone 5 commit `7061038`, with green 54-test remote CI.
- Model: existing 12,852,547-parameter primary U-Net.
- Training: AdamW, learning rate `2e-4`, batch 128, EMA `0.9999`, 50k-step
  initial ceiling.
- Gates: 20-step smoke, two-job 250+250 resume preflight, then one 50k run.
- Sampling: identical fixed initial tensors at 10k, 25k, and 50k.
- Execution: Hellbender through Slurm only.
- Authoritative plan: `docs/plans/full_cifar10_training.md`.

## Try index

| Try | Status | Configuration | Conclusion |
|---|---|---|---|
| `try01` | `READY` | Primary U-Net, batch 128, linear DDPM, max 50k | Implementation validated; execution pending |

## Current conclusion

No real-data optimization has run. Component-level evidence does not yet prove
that the complete system produces recognizable samples.

## Immediate next action

Publish the exact run commit, pull it to Hellbender, then submit the Slurm
environment/check gates and Gate A.

## Evidence

- Plan: `docs/plans/full_cifar10_training.md`
- Hypothesis: `Experiments/exp004-full-cifar-10-ddpm-training/hypothesis.md`
- Try: `Experiments/exp004-full-cifar-10-ddpm-training/try01/report.md`
- Milestone report: `Reports/milestone_06.md`
