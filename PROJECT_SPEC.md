# Diffusion Models from Scratch: Project Specification

## Objective

Build a small, readable DDPM implementation in PyTorch while deriving every
important equation and testing every transformation. CIFAR-10 is the first
dataset because its 32×32 RGB images make iteration inexpensive and visual
inspection straightforward.

## Learning goals

1. Derive and implement the forward noising process.
2. Implement the reverse-process parameterization and sampling algorithm.
3. Build the denoising network without importing a diffusion framework.
4. Train and evaluate a baseline DDPM reproducibly on CIFAR-10.
5. Connect each implementation step to its mathematical definition.

## Initial milestone

The first milestone establishes the repository and the data/theory foundation:

- a conventional `src/` Python package;
- a deterministic CIFAR-10 loader whose tensors are scaled from `[0, 1]` to
  `[-1, 1]`;
- an exact inverse transform for visualization;
- a seeded script that saves a clean grid of training examples;
- a derivation of `q(x_t | x_0)` for the DDPM forward process;
- a linear beta schedule with precomputed DDPM coefficients;
- a closed-form `q_sample` implementation;
- algebraic recovery of `x_start` when the exact noise is known; and
- independent hand-calculation, round-trip, and visual checks of the forward
  process.

## Scientific conventions and invariants

- Image layout is `(N, C, H, W)` and CIFAR-10 shape is `(N, 3, 32, 32)`.
- Model-space images lie in `[-1, 1]`.
- Visualization must apply the inverse transform and clamp only at the final
  display boundary.
- Randomness is controlled by explicit integer seeds; the initial seed is `0`.
- Timesteps use the mathematical indexing `t in {1, ..., T}`. Code may use
  zero-based array positions, but must document the mapping. In code, index `0`
  applies the first beta and is therefore close to, but not exactly, `x_start`.
- The milestone-one schedule is linear with `T=1000`, `beta_start=1e-4`, and
  `beta_end=2e-2`.
- Raw datasets live under `data/` and are not committed.
- Generated evidence is saved under `figures/` or a future numbered experiment
  try; completed experiment outputs will never be overwritten.

## Planned package structure

```text
src/diffusion_models/
  __init__.py
  data.py                 # CIFAR-10 transforms, loaders, inverse normalization
  diffusion/
    __init__.py
    schedules.py          # beta, alpha, cumulative alpha, coefficient extraction
    ddpm.py               # q_sample and exact-noise algebraic reconstruction
scripts/
  generate_cifar10_grid.py
  generate_forward_process_grid.py
tests/
  test_data.py
  test_forward_diffusion.py
docs/
  ddpm_forward_process.md
configs/                  # future training configurations
```

## DDPM/DDIM MVP roadmap

1. **Validated DDPM forward process:** dataset utilities, derivation, linear
   schedule, `q_sample`, exact-noise inversion, and a small empirical check.
2. **Time-conditioned U-Net:** readable CIFAR-10 denoiser with residual blocks,
   timestep conditioning, skip connections, and attention at 16×16.
3. **Training and tiny-subset overfit:** epsilon-prediction objective and a
   fixed, reproducible tiny-data integration test.
4. **EMA and checkpoint-resume:** complete training-state persistence and an
   exact deterministic resume test.
5. **DDPM sampling:** ancestral reverse process and sample trajectories.
6. **DDIM and runtime comparison:** deterministic reduced-step sampling using
   the same trained denoiser.
7. **Recruiter-facing presentation:** reproduction commands, figures, results,
   limitations, and interview-oriented explanations.

The canonical MVP is a tested PyTorch implementation of DDPM and DDIM on
CIFAR-10 with reproducible training, checkpointing, sampling, and a controlled
sampling-cost comparison.

## Deferred extensions

The following are explicitly outside the MVP: EDM, flow matching, class
conditioning, distributed training, large-scale FID sweeps, additional
datasets, deployment, and web interfaces.

## Current milestone: DDPM reverse sampling

Milestone 4 is published at commit `2f13b3f` with green remote CI. Milestone 5
is complete locally under `docs/plans/ddpm_reverse_sampling.md` and
`EXP003/try01`: exact posterior arithmetic, epsilon-to-clean reconstruction,
final-step noise suppression, seeded ancestral sampling, and one finite
1,000-call random-model trajectory are validated. This establishes structural
sampling correctness only; no trained-checkpoint or image-quality evaluation
has been performed.

## Initial acceptance criteria

- Installing with `pip install -e '.[dev]'` exposes `diffusion_models`.
- A CIFAR-10 batch has shape `(N, 3, 32, 32)` and lies in `[-1, 1]`.
- Inverse normalization reconstructs the original `[0, 1]` tensor within
  floating-point tolerance.
- Running the grid script with seed `0` produces a clean PNG with no random
  augmentation and correct colors.
- The forward-process note derives both the one-step transition and the closed
  form marginal, including assumptions and a reparameterized sampling formula.
- A three-step float64 schedule matches hand-computed alpha and cumulative-alpha
  values.
- `q_sample` matches an independently hand-computed two-image example.
- Forward noising followed by exact-noise inversion reconstructs `x_start` to
  `rtol=1e-10` and `atol=1e-10` across early, middle, and late timesteps.
- A visual check uses one normalized CIFAR-10 image and one fixed noise tensor at
  timesteps `[0, 100, 250, 500, 750, 999]`.

## Non-goals for the initial milestone

- Training a denoiser or submitting GPU/cluster jobs.
- Importing a diffusion implementation from another library.
- Claiming model quality before a frozen evaluation specification exists.
