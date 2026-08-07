# Hypothesis

## Primary hypothesis

The unchanged smoke U-Net can reduce epsilon-prediction MSE on both a fixed
synthetic noising problem and repeated stochastic noising of 16 fixed CIFAR-10
training images.

## Motivation and predicted result

Forward diffusion and the denoiser's gradient connectivity are already
validated independently. Correctly coupling the same sampled noise to `q_sample`
and the MSE target should therefore allow AdamW to reduce early-to-late mean
loss by at least 20% in each frozen gate.

## Alternative explanations

- Gate A could pass through simple memorization while Gate B exposes incorrect
  fresh-noise or per-image timestep handling.
- Loss could fall due to target leakage or accidentally reused noise rather
  than correct stochastic training; explicit batch construction and tests
  control this.
- Individual losses can fluctuate even when optimization works, so acceptance
  uses window means rather than monotonicity.

## Required controls

- Fixed inputs, timesteps, and noise in Gate A.
- Fresh independently sampled timesteps and noise in Gate B.
- Exact equality between the noise used by `q_sample` and the training target.
- Fixed 16-image manifest and disabled data augmentation.
- Finite-loss/gradient checks and proof that parameters changed.

## Falsification and interpretation

The hypothesis is falsified for `try01` if either gate misses its frozen 20%
windowed reduction, produces a nonfinite value, or fails to update parameters.
Passing supports only small-dataset pipeline learnability, not generation
quality or generalization.

## Stop conditions

Stop at the first failed gate. Do not alter the threshold, add steps, or change
the model within this try.

## Preregistration status

Frozen before implementation and execution on 2026-08-06 in
`docs/plans/training_and_overfit.md`.
