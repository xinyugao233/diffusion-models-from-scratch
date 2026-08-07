# EXP003 Try 01: DDPM Reverse Sampling

## Goal

Validate the posterior equations, one ancestral reverse step, and a complete
1,000-step seeded CPU chain through an EMA-populated random smoke U-Net.

## Hypothesis tested

The implementation will match independent posterior arithmetic, add no noise
at code timestep zero, preserve shape and finiteness, call the model exactly
1,000 times, and reproduce exactly under a fixed seed.

## Changes

Added the reverse-sampling module, exports, 13 focused tests, frozen smoke
config, derivation, and reproducible artifact script. No training, U-Net, or
forward-equation changes were made.

## Configuration

Authoritative plan: `docs/plans/ddpm_reverse_sampling.md`. Frozen values are
CPU float32, one thread, random smoke U-Net, EMA-copied weights, model seed 71,
sampling seed 73, and linear T=1000.

## Verification

All 54 repository tests passed. Ruff lint, Ruff formatting, and
`git diff --check` passed. Independent two-step float64 arithmetic matches the
posterior implementation, and completed Milestone 1/2 code paths are unchanged.

## Execution

Executed:

```bash
.venv/bin/python scripts/run_sampling_smoke.py \
  --config configs/sampling_smoke.json
```

The CPU inference-only run completed normally in 12.6002 seconds.

## Results

`PASS`: 1,000/1,000 model calls, finite `[4,3,32,32]` final tensor, and all six
trajectory keys `[1000,750,500,250,100,0]`. One summary, one final tensor, and
one trajectory PNG were produced as expected.

## Figures and tables

The 632×454 RGB four-row, six-column trajectory PNG was visually inspected.
Labels are correct and legible; the noise-like images are consistent with the
documented random-weight debug condition.

## Observations

The final state lies in `[-0.9998341,0.9998341]`. Every captured state is
finite. The independent artifact audit matched tensor, file, configuration,
figure, and source hashes and confirmed trajectory key zero equals the saved
final tensor.

## Failure analysis

No experimental failure occurred. The principal misuse risk is interpreting a
random-model debug visualization as an image-quality result; documentation
explicitly forbids that claim.

## Interpretation

The evidence supports structural correctness of posterior arithmetic,
timestep-zero handling, seeded stochastic execution, loop count, finiteness,
and EMA-model compatibility. It does not support generation-quality claims.

## Limitations

No trained checkpoint, dataset, FID, likelihood, DDIM comparison, GPU, or full
training is in scope.

## Next step

Review and commit this milestone before any trained-checkpoint evaluation or
DDIM implementation.

## Exact evidence paths

- Plan: `docs/plans/ddpm_reverse_sampling.md`
- Configuration: `configs/sampling_smoke.json`
- Summary: `Experiments/exp003-ddpm-reverse-sampling/try01/results/sampling_summary.json`
- Figure: `Experiments/exp003-ddpm-reverse-sampling/try01/figures/random_model_ddpm_trajectory.png`
