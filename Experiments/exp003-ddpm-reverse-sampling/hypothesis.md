# EXP003 Hypothesis: DDPM Reverse Sampling

## Primary hypothesis

The derived fixed-variance DDPM posterior, combined with the existing
epsilon-prediction interface, produces a shape-preserving, finite, seeded, and
fully reproducible ancestral reverse chain whose final code-timestep-zero step
adds no Gaussian noise.

## Motivation

Forward-process and denoiser tests cannot detect a wrong posterior coefficient,
an off-by-one reverse loop, or accidental final-step noise. These properties
must be established independently before interpreting any trained samples.

## Predicted result

Hand calculations and implementation agree in float64; supplied reverse noise
determines nonfinal samples exactly; timestep zero returns the mean; a seeded
1,000-step smoke run calls the model 1,000 times and remains finite.

## Alternative explanations

- A passing end-to-end smoke run alone could hide compensating coefficient
  errors, so hand arithmetic is required.
- Reproducibility could arise from a sampler that accidentally ignores all
  noise, so a different-seed control is required.
- A finite random-model output is only a numerical smoke check, not evidence of
  learned generation.

## Required controls

- Independent tiny-schedule posterior arithmetic.
- Supplied reverse-noise comparisons at nonzero and zero timesteps.
- Equal-seed and different-seed complete-loop comparisons.
- Exact model-call count and EMA-model interface check.

## Falsification conditions

Any coefficient mismatch, negative/nonfinite posterior variance, noise effect
at timestep zero, missing/extra loop call, nonfinite result, equal-seed
mismatch, or identical different-seed output falsifies the try.

## Interpretation rules

Passing establishes structural sampler correctness only. It does not establish
sample quality, checkpoint quality, likelihood, FID, or generalization.

## Stop conditions

Stop on a failed mathematical gate or nonfinite full-chain state. Do not weaken
the frozen criteria or begin training to rescue the result.

## Preregistration status

Frozen on 2026-08-07 before implementation and execution of `EXP003/try01`.
