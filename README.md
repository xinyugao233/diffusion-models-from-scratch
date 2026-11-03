# Diffusion Models from First Principles

A from-scratch PyTorch implementation of DDPM and DDIM on CIFAR-10, covering
forward diffusion, a custom time-conditioned U-Net, epsilon-prediction
training, EMA and exact checkpoint resume, ancestral generation, and
accelerated deterministic sampling.

**Engineering highlights:** custom 12.85M U-Net; DDPM/DDIM samplers; EMA and
checkpoint recovery; 83 CPU tests; portable inference exports. In the fixed
16-seed H100 benchmark, DDIM-50 used 50 calls and took 0.225 s versus
4.611 s for DDPM-1000 (20.52×); image quality was assessed qualitatively.

[Run the CPU demo](#cpu-quickstart) · [Browse the implementation](#source-reading-guide)
· [Inspect benchmark evidence](#accelerating-sampling-with-ddim)

## Generated Samples After 50K Training Steps

The 12.85M-parameter model was trained for 50,000 steps on the full CIFAR-10
training split. This fixed-seed grid uses the final EMA checkpoint and the
validated 1,000-step ancestral DDPM sampler.

![Generated CIFAR-10 samples from the 50K-step EMA checkpoint](Experiments/exp004-full-cifar-10-ddpm-training/try01/figures/full/ema_samples_step_050000.png)

## CPU quickstart

Python 3.13 is the tested local review interpreter. Installation requires network
access; this synthetic CPU demo needs no dataset, GPU, or checkpoint afterward.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -c constraints-review.txt -e '.[dev]'
python scripts/run_synthetic_optimization.py --config configs/public_cpu_demo.json --output-dir "${DEMO_OUTPUT:-outputs/cpu-demo}"
```

Expected: `passed: true`, 39 epsilon-prediction updates plus one attention
fixture update (40 total), and `metrics.jsonl` / `summary.json` at the printed
output location. The optimizer loop took **1.86 s** on the recorded local
CPU review environment (startup excluded). Repeating into the same directory
fails before optimization;
set `DEMO_OUTPUT=outputs/cpu-demo-2` for a new run. Historical configs and tries
are unchanged. See [measured runtime and checks](docs/maintenance/verification.md).

## Portable trained inference

The original 50K checkpoint was recovered and its recorded SHA-256 verified.
An EMA-only inference bundle is staged locally for review; **there is no public
weights download yet**. With an authorized local bundle:

```bash
python scripts/sample.py --checkpoint /path/to/ema-50000-v1.pt --output-dir outputs/samples-01 --device cpu --sampler ddim --inference-steps 50 --sample-count 16 --seed 0
```

No Slurm or optimizer state is required for this interface. CUDA is optional.
Missing/incompatible weights and existing output directories are rejected.
[Export, model card and availability](docs/maintenance/inference.md).

## Training Progression

Using the same fixed initial-noise seeds at each checkpoint, the samples
progress from coarse texture at 10K steps to color-separated silhouettes at
25K and recognizable CIFAR-like structure at 50K. The final training loss was
`0.033687`, and the last-1,000/first-1,000 mean-loss ratio was `0.465576`.

![Fixed-seed EMA progression at 10K, 25K, and 50K steps](Experiments/exp004-full-cifar-10-ddpm-training/try01/figures/full/fixed_seed_progression_010000_025000_050000.png)

## Accelerating Sampling with DDIM

The same 50K-step EMA denoiser was evaluated with ancestral DDPM and
deterministic DDIM (`eta=0`). Every condition started from the same materialized
batch of 16 initial Gaussian tensors; the four columns below are aligned by
initial-noise seed.

![Paired DDPM-1000 and DDIM-100, DDIM-50, and DDIM-25 samples](figures/presentation_v2/sampling_comparison.png)

| Sampler | NFE | Median H100 runtime | Speedup |
|---|---:|---:|---:|
| DDPM | 1000 | `4.611 s` | `1.00×` |
| DDIM | 100 | `0.450 s` | **`10.24×`** |
| DDIM | 50 | `0.225 s` | **`20.52×`** |
| DDIM | 25 | `0.112 s` | **`41.03×`** |

Deterministic DDIM reused the same 50K-step EMA denoiser and reduced sampling
to 25–100 network evaluations, yielding measured 10.2×–41.0× H100 speedups
while retaining recognizable CIFAR-like structure in the fixed 16-seed
qualitative comparison.

This visual comparison is not FID/KID or distribution-level image-quality
evidence. The timing result is specific to this checkpoint, implementation,
batch, hardware, and synchronized three-repeat protocol. DDPM is intrinsically
stochastic but repeated bitwise under a frozen reverse RNG seed; eta-zero DDIM
requires no reverse-step random draws and is deterministic conditional on the
initial `x_T`.

## How the System Fits Together

```mermaid
flowchart TD
    A["CIFAR-10 clean image x₀"] --> B["Forward diffusion q(xₜ | x₀)"]
    B --> C["Noisy image xₜ + timestep t"]
    C --> D["Time-conditioned U-Net"]
    D --> E["Predict noise εθ(xₜ, t)"]
    E --> F["Epsilon-prediction training"]
    F --> G["50K-step EMA denoiser"]
    G --> H["DDPM: 1,000 NFE"]
    G --> I["DDIM: 25–100 NFE"]
    H --> J["Generated x₀"]
    I --> J
```

## What the Project Implements

- A linear DDPM noise schedule and closed-form forward sampler
  `q(x_t | x_0)`.
- A custom time-conditioned U-Net with residual blocks, skip connections, and
  attention at 16x16 resolution.
- Epsilon-prediction training on CIFAR-10 with AdamW and exponential moving
  average weights.
- Versioned checkpoints that preserve the model, EMA, optimizer, global step,
  configuration, and Python/PyTorch random state for exact resume.
- A 1,000-step ancestral DDPM sampler and an eta-zero DDIM sampler that reuses
  the same trained denoiser without retraining.
- Reproducible experiment configurations, fixed-seed visual comparisons,
  artifact hashes, and CPU-only continuous integration.

## Core Mathematics

With `alpha_t = 1 - beta_t` and
`bar_alpha_t = product_{s=1}^t alpha_s`, the forward process can sample any
timestep directly:

```text
x_t = sqrt(bar_alpha_t) x_0 + sqrt(1 - bar_alpha_t) epsilon,
epsilon ~ N(0, I).
```

The U-Net receives both `x_t` and `t` and learns to recover the sampled noise:

```text
L_simple = E[ || epsilon - epsilon_theta(x_t, t) ||^2 ].
```

DDPM uses the predicted noise to parameterize a stochastic ancestral reverse
transition. DDIM uses the same prediction function on a shorter trajectory;
with `eta=0`, its reverse update is deterministic conditional on `x_T` and a
deterministic evaluation-mode denoiser in the stated environment. Full
derivations are in:

- [DDPM forward process](docs/ddpm_forward_process.md)
- [DDPM reverse process](docs/ddpm_reverse_process.md)
- [DDIM sampling](docs/ddim_sampling.md)

## Reproduce and Validate

Create a local environment and run the inexpensive CPU checks:

```bash
python3.13 -m venv .venv
.venv/bin/pip install -c constraints-review.txt -e '.[dev]'
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

The CPU suite covers data transforms, diffusion equations, U-Net dtype behavior,
training, EMA/checkpoint restoration, explicit interrupted-log recovery,
DDPM/DDIM sampling, inference bundles, and figure regression. CI also runs the
literal installed public-demo entry point; see the
[hosted PR checks](https://github.com/xinyugao233/diffusion-models-from-scratch/pull/1/checks) for the current branch status. [Verification record](docs/maintenance/verification.md).

### Optional CIFAR-10 download examples

These commands download CIFAR-10 unless `--no-download` is given. They write
to ignored `outputs/` paths and refuse an existing file.

```bash
python scripts/generate_cifar10_grid.py
python scripts/generate_forward_process_grid.py
```

Completed experiment outputs are immutable. Reproducing a scientific run must
use a new numbered try and new output paths. Full CIFAR-10 training and the H100
runtime comparison are guarded Slurm workflows; see
[cluster/README.md](cluster/README.md).

## Evidence and Reproducibility

- [Full CIFAR-10 training report](Reports/milestone_06.md): exact checkpoint,
  configuration, loss gates, runtime, failures, and qualitative assessment.
- [DDPM/DDIM comparison report](Reports/milestone_07.md): frozen protocol,
  exact timings, hashes, visual inspection, provenance, and limitations.
- [Machine-readable DDIM comparison](Experiments/exp005-ddim-sampling-speed-quality-comparison/try01/results/comparison.json)
- [DDIM comparison manifest](Experiments/exp005-ddim-sampling-speed-quality-comparison/try01/results/manifest.json)
- [Project specification](PROJECT_SPEC.md)

The run persisted the fixed initial-noise seeds but not an explicit hash of the
materialized `x_T` batch. The exact implementation passed one in-memory batch
to every sampler condition; a batch hash was reconstructed locally from the
frozen generation function and seeds and is documented as derived—not
run-recorded—evidence.

## Source reading guide

| Topic | Implementation | Contract / checks |
|---|---|---|
| Forward diffusion and indexing | [ddpm.py](src/diffusion_models/diffusion/ddpm.py) | [Derivation](docs/ddpm_forward_process.md) |
| DDIM transitions and captures | [ddim.py](src/diffusion_models/diffusion/ddim.py) | [Analytic tests](tests/test_ddim.py), [state convention](docs/ddim_sampling.md) |
| Custom 12.85M U-Net | [unet.py](src/diffusion_models/models/unet.py) | [Model tests](tests/test_unet.py) |
| EMA and interrupted-run recovery | [recovery.py](src/diffusion_models/recovery.py) | [Recovery contract](docs/maintenance/recovery.md), [fixtures](tests/test_recovery_inference.py) |
| Portable inference | [inference.py](src/diffusion_models/inference.py) | [Release staging](docs/maintenance/inference.md) |
| Historical experiment evidence | [Experiments](Experiments/) | [Artifact inventory](docs/maintenance/artifacts.md) |

The [maintenance summary](docs/maintenance/verification.md) is the current public
maintenance status. Root planning notes remain historical research context;
[archive policy](docs/maintenance/history.md) explains preserved references.
No new license has been selected; publication rights require owner review.

## Repository Layout

```text
src/diffusion_models/   reusable DDPM, DDIM, U-Net, training, and checkpoint code
scripts/                reproducible local and experiment entry points
configs/                frozen architecture and run configurations
tests/                  CPU correctness and regression tests
docs/                   mathematical derivations and implementation notes
Experiments/            immutable experiment plans, tries, results, and reports
Reports/                milestone-level evidence summaries
cluster/                guarded Hellbender/Slurm execution workflows
```

## Scope

This project demonstrates a complete, tested DDPM/DDIM learning pipeline on
CIFAR-10 and a controlled checkpoint-specific sampling-speed comparison. It
does not establish FID/KID parity, likelihood, held-out generalization,
cross-hardware performance, or universal DDIM quality at low step counts.
