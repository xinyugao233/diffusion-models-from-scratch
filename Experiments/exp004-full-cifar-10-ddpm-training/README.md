# EXP004 Full CIFAR-10 DDPM Training

## Status

`COMPLETED` — preflight gates and the frozen 50,000-step run passed on
Hellbender; outputs and figures were validated.

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
| `try01` | `COMPLETED` | Primary U-Net, batch 128, linear DDPM, 50k | Stable 50k training, checked resume, and recognizable fixed-seed EMA samples |

## Current conclusion

All 50,000 loss records were finite and contiguous. The last-1,000/first-1,000
mean-loss ratio was `0.465576`, passing the frozen `0.9` maximum. The 10k,
25k, and 50k fixed-seed grids show increasing structure, with recognizable
CIFAR-like animal and vehicle forms at 50k. This supports only the planned
end-to-end baseline claim; no FID, comparative, or DDIM claim is made.

## Immediate next action

Close Milestone 6. Any 100k extension or DDIM work requires a separate reviewed
plan and is not authorized by this try.

## Evidence

- Plan: `docs/plans/full_cifar10_training.md`
- Hypothesis: `Experiments/exp004-full-cifar-10-ddpm-training/hypothesis.md`
- Try: `Experiments/exp004-full-cifar-10-ddpm-training/try01/report.md`
- Milestone report: `Reports/milestone_06.md`
- Fixed-seed progression:
  `Experiments/exp004-full-cifar-10-ddpm-training/try01/figures/full/fixed_seed_progression_010000_025000_050000.png`
