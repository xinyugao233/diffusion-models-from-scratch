# EXP002 Deterministic Checkpoint Resume

## Status

`COMPLETED` — try01 preserves the comparator failure and corrected try02 passes.

## Research question

Does saving and restoring model, EMA, AdamW, step, configuration, and RNG state
preserve the exact deterministic CPU training trajectory?

## Motivation and program connection

Milestone 3 establishes learnability. This experiment validates reliable
training-state persistence before any long run or reverse sampler is added.

## Design

- Baseline: uninterrupted 100-step smoke-U-Net training.
- Intervention: checkpoint at step 50, destroy/reconstruct objects, restore,
  and continue to step 100.
- Negative control: omit RNG restoration and require the next stochastic batch
  to differ.
- Authoritative plan: `docs/plans/ema_and_checkpoints.md`.

## Try index

| Try | Status | Configuration | Conclusion |
|---|---|---|---|
| `try01` | `FAILED` | CPU, float32, AdamW, EMA 0.99, 100/50 steps | Comparator rejected equivalent mapping subclasses |
| `try02` | `COMPLETED` | Same scientific configuration; corrected comparator | All 12 exact comparisons passed |

## Current conclusion

On the frozen local CPU stack, uninterrupted 100-step training and 50+50
checkpoint/resume training are exactly equal in model, EMA, AdamW, full loss
trajectory, global step, and next random draw/loss. The omission negative
control proves RNG restoration is necessary.

## Immediate next action

Stop at Milestone 4 and await review or explicit authorization for DDPM
sampling.

## Evidence

- Plan: `docs/plans/ema_and_checkpoints.md`
- Hypothesis: `Experiments/exp002-deterministic-checkpoint-resume/hypothesis.md`
- Try: `Experiments/exp002-deterministic-checkpoint-resume/try01/report.md`
- Best try: `Experiments/exp002-deterministic-checkpoint-resume/try02/report.md`
- Milestone report: `Reports/milestone_04.md`
