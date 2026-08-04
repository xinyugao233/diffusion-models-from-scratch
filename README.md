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

## Layout

- `src/diffusion_models/`: reusable implementation
- `scripts/`: reproducible entry points
- `tests/`: inexpensive correctness tests
- `docs/`: mathematical derivations
- `configs/`: future frozen run configurations
- `figures/`: generated visual checks

No training experiment has been defined or run yet.
