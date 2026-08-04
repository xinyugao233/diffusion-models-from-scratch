# Milestone 01 Report: Data and DDPM Forward Process

## Objective

Establish a reproducible CIFAR-10 input pipeline and implement the mathematical
foundation of DDPM forward noising before any denoiser or training code.

## Implemented

- CIFAR-10 loading and deterministic `[-1,1]` normalization.
- Exact inverse normalization for CHW and NCHW tensors.
- Seeded 8×8 dataset grid.
- Derivation of the one-step transition and closed-form `q(x_t | x_0)`.
- Linear schedule with `T=1000`, `beta_start=1e-4`, `beta_end=2e-2`.
- Precomputed alpha, cumulative-alpha, and square-root coefficients.
- Per-example timestep extraction with broadcast-safe shapes.
- Closed-form `q_sample` and exact-noise reconstruction.
- Fixed-image, fixed-noise CIFAR-10 forward-process visualization.

## Scientific invariants

- Model-space CIFAR-10 tensors use `[-1,1]`.
- Code timestep `0` applies the first beta; it is not an identity transition.
- The forward visualization uses image index `0`, seed `0`, and the same noise
  tensor at timesteps `[0,100,250,500,750,999]`.
- No training, checkpointing, or model-quality evaluation was performed.

## Verification

```text
.venv/bin/pytest -q tests/test_forward_diffusion.py  -> 3 passed
.venv/bin/pytest -q                                 -> 7 passed
.venv/bin/ruff check .                              -> all checks passed
.venv/bin/ruff format --check .                     -> 19 files already formatted
```

The forward tests independently verify hand-computed schedule values,
hand-computed forward samples, and float64 algebraic recovery at early, middle,
and late timesteps. Both generated figures were visually inspected.
The observed maximum absolute float64 reconstruction error was
`4.873879078104437e-14`.

## Evidence

- `tests/test_data.py`
- `tests/test_forward_diffusion.py`
- `figures/cifar10_train_grid_seed0.png`
- `figures/cifar10_forward_process_seed0.png`
- `docs/ddpm_forward_process.md`

## Result and interpretation

The requested milestone implementation passes its deterministic mathematical
and code-quality checks. The evidence supports correctness of the tested
closed-form computations; it does not yet establish empirical distributional
agreement, denoising ability, or model quality.

## Next step

Before training, add empirical mean/variance validation for `q_sample`, then
specify the timestep embedding and denoiser architecture as a new milestone.
