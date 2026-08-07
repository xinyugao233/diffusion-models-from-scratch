# EXP001 Epsilon Prediction Learnability

## Status

`COMPLETED` — both frozen gates passed and all output records were validated.

## Research question

Can the validated smoke U-Net optimize the DDPM epsilon-prediction objective on
a fixed synthetic problem and a fixed 16-image CIFAR-10 subset?

## Motivation and program connection

Milestones 1 and 2 validate forward diffusion and the denoiser architecture,
respectively. This experiment is the smallest end-to-end learnability gate
before implementing training infrastructure or reverse sampling.

## Design

- Baseline: committed Milestone 2 state with no optimizer updates.
- Intervention: explicit epsilon-prediction MSE and AdamW updates.
- Controls: fixed synthetic batch for Gate A; fixed unaugmented 16-image subset
  for Gate B.
- Scientific invariants and acceptance thresholds are frozen in
  [`docs/plans/training_and_overfit.md`](../../docs/plans/training_and_overfit.md).

## Try index

| Try | Status | Configuration | Conclusion |
|---|---|---|---|
| `try01` | `COMPLETED` | Smoke U-Net, linear schedule, CPU | Both learnability gates passed |

## Current conclusion

The explicit epsilon-prediction pipeline reduced late-window mean loss by
98.93% relative to the early window on the fixed synthetic batch and by 80.07%
on repeated stochastic noising of 16 fixed CIFAR-10 images. This supports
small-dataset pipeline learnability only.

## Immediate next action

Stop at Milestone 3 and await review or explicit authorization for the next
roadmap stage.

## Evidence

- Authoritative plan: `docs/plans/training_and_overfit.md`
- Hypothesis: `Experiments/exp001-epsilon-prediction-learnability/hypothesis.md`
- Try record: `Experiments/exp001-epsilon-prediction-learnability/try01/report.md`
- Milestone report: `Reports/milestone_03.md`
- Loss curve: `Experiments/exp001-epsilon-prediction-learnability/try01/figures/overfit_loss_curve.png`
