# EXP005 Hypothesis: DDIM Sampling Speed–Quality Comparison

## Core hypothesis

Using the same 50k EMA epsilon predictor and initial Gaussian tensors,
deterministic DDIM trajectories of 100, 50, and 25 model evaluations will be
substantially faster than 1,000-step ancestral DDPM sampling while retaining
varying degrees of recognizable CIFAR-like structure.

## Expected mechanism

DDIM uses the same trained model but traverses a selected timestep subsequence.
At `eta=0`, the reverse update injects no new noise, so each initial `x_T`
deterministically maps to one output. Runtime should scale primarily with the
number of U-Net evaluations.

## Predicted result

- Exact evaluation counts of 1,000, 100, 50, and 25.
- Every setting finite and exactly reproducible across three repeats.
- DDIM medians faster than DDPM-1000, approximately tracking evaluation count.
- DDIM-100/50 retain more visible structure than DDIM-25.

## Alternative explanations and controls

GPU warm-up or asynchronous timing could create an artificial speedup; a
shared warm-up and synchronized timers control this. Different seeds could
create apparent quality differences; every grid uses the same fixed initial
tensors. The stochastic DDPM path is not expected to match DDIM pixelwise even
from the same initial noise.

## Falsification conditions

Incorrect call counts, nonfinite outputs, nondeterministic DDIM repeats,
checkpoint/config drift, or DDIM medians not faster than DDPM falsify the
corresponding implementation or speed claim. Poor 25-step images do not
falsify correct deterministic sampling; they demonstrate the tradeoff.

## Interpretation rules

The result is a fixed-checkpoint, fixed-seed, single-H100 comparison. No FID,
distribution-level quality, generalization, or universal speed claim is
allowed.

## Stop conditions

Follow `plan.md`; stop after the four frozen settings. Do not add training,
100k extension, more samplers, or more step counts.

## Preregistration status

Frozen on 2026-08-12 before DDIM implementation and execution.
