# Diffusion Models from Scratch

A compact PyTorch project for learning diffusion models by deriving and
implementing a DDPM one component at a time. The initial target is CIFAR-10.

The governing design and acceptance criteria are in [PROJECT_SPEC.md](PROJECT_SPEC.md).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

## Generate the CIFAR-10 grid

```bash
.venv/bin/python scripts/generate_cifar10_grid.py
```

This downloads CIFAR-10 into `data/` when needed and writes
`figures/cifar10_train_grid_seed0.png`.

## Generate the forward-process grid

```bash
.venv/bin/python scripts/generate_forward_process_grid.py
```

This applies one fixed noise tensor to one normalized CIFAR-10 image at code
timesteps `0, 100, 250, 500, 750, 999` and writes
`figures/cifar10_forward_process_seed0.png`.

The schedule and forward-process API live under
`diffusion_models.diffusion`. Mathematical details and the code-index mapping
are in [docs/ddpm_forward_process.md](docs/ddpm_forward_process.md).

## Reproduce the learnability gates

```bash
.venv/bin/python scripts/run_synthetic_optimization.py \
  --config configs/synthetic_optimization.json

.venv/bin/python scripts/run_overfit_16.py \
  --config configs/overfit_16.json \
  --download
```

These immutable experiment outputs already exist in `EXP001/try01`, so reruns
must use a new numbered try and updated output paths. The completed result and
its limited scientific interpretation are documented in
[Reports/milestone_03.md](Reports/milestone_03.md).

## Layout

- `src/diffusion_models/`: reusable implementation
- `scripts/`: reproducible entry points
- `tests/`: inexpensive correctness tests
- `docs/`: mathematical derivations
- `configs/`: frozen architecture and run configurations
- `figures/`: generated visual checks

Milestone 3 verifies only that the epsilon-prediction pipeline can learn a
fixed 16-image dataset. Reverse sampling and generation quality are not yet
implemented or evaluated.

## Validate exact checkpoint resume

Milestone 4's corrected immutable result was produced with:

```bash
.venv/bin/python scripts/run_resume_validation.py \
  --config configs/resume_validation_try02.json
```

The existing `try01` and `try02` outputs must not be overwritten; a reproduction
must copy the frozen settings into a new numbered try with new output paths.
EMA, checkpoint schema, the preserved comparator failure, exact corrected
result, and limitations are documented in
[Reports/milestone_04.md](Reports/milestone_04.md).
