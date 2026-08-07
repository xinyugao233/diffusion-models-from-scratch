# EXP003 DDPM Reverse Sampling

## Status

`COMPLETED` — try01 passed all mathematical, unit, artifact, and full-chain
smoke gates.

## Research question

Does the DDPM posterior and ancestral reverse loop match independently checked
mathematics, handle the deterministic final step correctly, and run a seeded
1,000-step chain through the existing U-Net interface?

## Motivation and program connection

Milestones 1–4 establish forward diffusion, the denoiser architecture,
epsilon-prediction learnability, EMA, and exact checkpoint resume. This
experiment supplies the remaining reverse transition while keeping sampling
structural correctness separate from generation quality.

## Design

- Baseline: Milestone 4 commit `2f13b3f` without reverse sampling.
- Intervention: fixed-variance DDPM posterior, one-step ancestral sampler, and
  complete reverse loop.
- Controls: independent tiny-schedule arithmetic, supplied reverse noise,
  timestep-zero noise masking, seed reproducibility, and model-independent EMA
  use.
- Full smoke: one random-weight EMA smoke U-Net, CPU, 1,000 steps, four images.
- Authoritative plan: `docs/plans/ddpm_reverse_sampling.md`.

## Try index

| Try | Status | Configuration | Conclusion |
|---|---|---|---|
| `try01` | `COMPLETED` | CPU, random EMA smoke U-Net, linear T=1000 | Exact posterior gates and finite 1,000-call chain passed |

## Current conclusion

The posterior matches independent arithmetic, the last step adds no noise, and
the seeded random EMA smoke model completed a finite 1,000-call chain with the
requested six-state trajectory. This establishes structural sampling
correctness only, not image quality.

## Immediate next action

Review and commit Milestone 5 before any trained-checkpoint sampling or DDIM
work.

## Evidence

- Plan: `docs/plans/ddpm_reverse_sampling.md`
- Hypothesis: `Experiments/exp003-ddpm-reverse-sampling/hypothesis.md`
- Try: `Experiments/exp003-ddpm-reverse-sampling/try01/report.md`
- Milestone report: `Reports/milestone_05.md`
