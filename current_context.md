# Current Context

The repository's first milestone is defined in `PROJECT_SPEC.md`. The package,
CIFAR-10 loader, inverse normalization, grid script, tests, and forward-process
derivation are implemented. A local `.venv` contains the declared dependencies.
The linear schedule, coefficient extraction, closed-form forward sampler, and
exact-noise reconstruction are also implemented. All seven unit tests and Ruff
pass. Deterministic seed-0 dataset and forward-process grids were generated
using the existing CIFAR-10 copy at
`../diffusion-memorization-geometry/data/cifar10` and visually inspected. The
forward grid fixes image index 0 and one noise realization across code
timesteps `[0,100,250,500,750,999]`.
