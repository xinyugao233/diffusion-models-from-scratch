# Milestone 05 Report: DDPM Reverse Sampling

## Objective

Implement and validate the fixed-variance DDPM posterior, one ancestral reverse
step, and a complete seeded reverse chain using the existing epsilon-prediction
U-Net and model-independent EMA interface.

## Baseline

- Baseline commit: `2f13b3f785b6adfd815199b5a6274dd6b5245cee`
- Milestone 4 GitHub Actions: run `31152749677`, SUCCESS, 46 seconds
- Remote checks: 41 tests, Ruff lint, and Ruff formatting passed
- Milestone 5 execution state: local and uncommitted

## Implemented behavior

- Exact posterior variance and both posterior-mean coefficients with
  `bar(alpha)_0=1`.
- Reuse of `predict_x_start_from_noise` for epsilon-to-clean reconstruction.
- Optional predicted-clean clipping to model space `[-1,1]`, enabled by
  default and frozen for the smoke run.
- One-step sampling with explicit supplied-noise or seeded-generator paths.
- Deterministic code-timestep-zero output with no random-number consumption.
- Mixed-timestep batch masking.
- Complete `T`-call reverse loop with per-step finite checks and selected state
  capture.
- Model-independent EMA use by copying EMA state into an ordinary model.

The existing U-Net, forward sampler, and schedule implementation were not
changed.

## Mathematical and unit verification

The new 13-test sampling suite includes independent two-step float64 posterior
arithmetic, timestep-zero coefficients, broadcasting, a known epsilon model,
manual supplied-noise sampling, mixed-step masking, call count, same/different
seed controls, a random smoke U-Net, EMA compatibility, and invalid-input
errors.

```text
.venv/bin/python -m pytest -q
  -> 54 passed
.venv/bin/ruff check .
  -> All checks passed
.venv/bin/ruff format --check .
  -> 59 files already formatted
git diff --check
  -> passed with no output
```

Normal tests are CPU-only, dataset-free, and use short schedules.

## EXP003/try01 full-chain smoke result

```text
Command: .venv/bin/python scripts/run_sampling_smoke.py \
         --config configs/sampling_smoke.json
Model: random smoke U-Net copied through EMA, 491,107 parameters
Model seed / sampling seed: 71 / 73
Shape: [4,3,32,32]
Schedule: linear T=1000, beta 1e-4 to 2e-2
Device/dtype: one Apple arm64 CPU thread / float32
Model calls: 1000 / 1000 expected
Captured states: [1000,750,500,250,100,0]
Duration: 12.6002 seconds
Final finite: true
Final range: [-0.9998341, 0.9998341]
Result: PASS
```

The saved trajectory is a 632×454 RGB PNG with four rows and six correctly
labeled columns. It was visually inspected. The noise-like appearance is
expected from random weights and is not interpreted as generation quality.

## Independent artifact audit

The summary was parsed separately; the final tensor was loaded independently;
shape, finiteness, tensor hash, file hash, configuration hash, figure hash,
model-call count, trajectory key count/order, and equality of the final tensor
hash with trajectory key zero were all checked. The audit passed.

- Configuration SHA-256:
  `5899bea12bef0a88e3e865ac4de86178f83fa4181eedbd5b815440ac83ee9de5`
- Sampling source SHA-256:
  `13afbe9e5a259c843aa57eaeed6ac7a5ac96a011b26dfd479138f1a8183741e5`
- Smoke script SHA-256:
  `af759d615cb310ab35cda41ac6a3671f4879df46cf6fbca290daa339df00a70b`
- Summary SHA-256:
  `0de6e664a015a7786258548575d81445f65a21f62249a3c18b158995cdee2c26`
- Final tensor file SHA-256:
  `a87eff3ad1497095641dab018ba3188a94483d7b0023530a8eca1e6f1aa93360`
- Figure SHA-256:
  `f3dfb2da67a9acbcb5ec7731355162c1511b3d5361a1df6700a21c003539485c`

## Scientific conclusion

The evidence supports this limited claim:

> The implemented fixed-variance DDPM posterior matches independent arithmetic,
> the final reverse step adds no noise, and the existing EMA-compatible U-Net
> interface can execute a finite, seeded, shape-correct 1,000-step ancestral
> chain with exactly 1,000 model calls.

It does not establish image quality, trained-checkpoint quality, CIFAR-10
generation, likelihood, FID, DDIM correctness, or sampling speed on GPU.

## Evidence paths

- Plan: `docs/plans/ddpm_reverse_sampling.md`
- Derivation: `docs/ddpm_reverse_process.md`
- Understanding: `docs/understanding/sampling.md`
- Try report: `Experiments/exp003-ddpm-reverse-sampling/try01/report.md`
- Summary: `Experiments/exp003-ddpm-reverse-sampling/try01/results/sampling_summary.json`
- Final tensor: `Experiments/exp003-ddpm-reverse-sampling/try01/results/final_sample.pt`
- Figure: `Experiments/exp003-ddpm-reverse-sampling/try01/figures/random_model_ddpm_trajectory.png`
