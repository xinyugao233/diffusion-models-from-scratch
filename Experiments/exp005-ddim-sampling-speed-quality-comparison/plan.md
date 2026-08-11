# EXP005 Plan: DDIM Sampling Speed–Quality Comparison

## Status and authority

This is the authoritative Milestone 7 plan. It was frozen on 2026-08-12 before
DDIM implementation or checkpoint evaluation. It uses the completed
`EXP004/try01` 50k EMA checkpoint and does not authorize training, checkpoint
extension, FID/KID infrastructure, or additional samplers.

## Research question

How much wall-clock sampling speed is gained by replacing the validated
1,000-step ancestral DDPM chain with deterministic DDIM trajectories of 100,
50, or 25 denoiser evaluations, and what visible structure is retained when
all methods start from the same 16 Gaussian tensors?

## Motivation and existing evidence

Milestone 6 established stable full-CIFAR-10 training and recognizable 50k EMA
samples under 1,000-step DDPM sampling. The original DDIM paper shows that the
same DDPM-trained epsilon predictor can be evaluated along a shorter
non-Markovian trajectory without retraining. This experiment isolates sampler
choice while freezing the trained denoiser and initial noise.

## Hypothesis and allowed interpretation

DDIM-100, DDIM-50, and DDIM-25 will require exactly 10%, 5%, and 2.5% of the
DDPM-1000 denoiser evaluations and will reduce median synchronized GPU runtime
substantially. Fewer steps may progressively reduce fine detail or coherent
structure.

Passing supports only a checkpoint-specific quality–compute observation for
the frozen 16 seeds. It does not establish distribution-level image quality,
FID/KID, a general DDIM speedup across hardware, or superiority over other
samplers.

## Baseline, intervention, and controls

- Baseline: validated `p_sample_loop`, 1,000 stochastic DDPM steps.
- Intervention: deterministic DDIM (`eta=0`) with 100, 50, or 25 selected code
  timesteps.
- Checkpoint control: exact 50k `EXP004/try01` checkpoint only.
- Model control: the 50k EMA weights copied into the unchanged primary U-Net.
- Noise control: fixed initial-noise seeds `1000..1015` for every setting.
- DDPM randomness: reverse-noise generator seed `109`, recreated for every
  timed repeat.
- DDIM randomness: no reverse noise; identical initial state determines output.
- Batch control: all 16 samples generated together in float32 on one H100.
- Postprocessing control: inverse normalization and clamping only for PNGs.
- No cherry-picking: all 16 fixed rows appear in every grid.

## Exact checkpoint identity

- Remote path:
  `/home/xggh8/data/diffusion-models-from-scratch/exp004-full-cifar10-ddpm-training/try01/full/checkpoints/checkpoint_step_050000.pt`
- Size: `205,996,222` bytes
- SHA-256:
  `f6b64f7028432ac1690029564610918e68e0c45b5188a97961863eadd79b17b9`
- Global step: `50,000`
- Training config: `configs/full_cifar10_training.json`
- Training config SHA-256:
  `8d4480ef52c2d299320cc374ae893b07a16690e091c88f303d1b9c8bf0140487`

## Mathematical definition

For current code timestep `t` and selected previous code timestep `s<t`, use
the training schedule cumulative products `alpha_bar[t]` and `alpha_bar[s]`.
For the final transition, define `alpha_bar[-1] = 1`.

The model predicts epsilon and the clean image estimate is

```text
x0_hat = (x_t - sqrt(1 - alpha_bar_t) * epsilon_theta(x_t,t))
         / sqrt(alpha_bar_t).
```

With the existing `clip_x_start=true` control, clamp only `x0_hat` to
`[-1,1]`. Deterministic DDIM uses `eta=0`, hence `sigma_t=0`, and

```text
x_s = sqrt(alpha_bar_s) * x0_hat
      + sqrt(1 - alpha_bar_s) * epsilon_theta(x_t,t).
```

No new Gaussian noise is drawn. The selected inference timesteps are rounded
linear spacing over code indices `[0,999]`, include both endpoints, are unique,
and are traversed in strictly descending order. The number of selected indices
equals the number of model evaluations.

## Implementation scope

- Add `src/diffusion_models/diffusion/ddim.py` with `ddim_timesteps`,
  `ddim_step`, and `ddim_sample_loop`.
- Export the API without refactoring the existing DDPM sampler.
- Add focused CPU tests with independent arithmetic and call-count checks.
- Add `docs/ddim_sampling.md` and interview notes under
  `docs/understanding/ddim_sampling.md`.
- Add one Slurm-only checkpoint comparison entrypoint and one guarded H100
  launcher.
- Add no training code and no third-party diffusion implementation.

## Evaluation protocol

| Setting | Model evaluations | Stochasticity |
|---|---:|---|
| DDPM-1000 | 1000 | fixed reverse seed 109 |
| DDIM-100 | 100 | deterministic (`eta=0`) |
| DDIM-50 | 50 | deterministic (`eta=0`) |
| DDIM-25 | 25 | deterministic (`eta=0`) |

After checkpoint loading, perform one untimed model warm-up. Run each complete
setting three times, recreating its fixed inputs each time. Synchronize CUDA
immediately before and after each trajectory. Timing excludes checkpoint I/O,
warm-up, tensor hashing, CPU transfer, and figure writing. Report all three
times and their median, total model evaluations, median images/s, and speedup
relative to DDPM-1000.

Every repeated output for a setting must be bitwise identical. Save the first
output tensor and a 4x4 grid. Construct one headline comparison containing all
four grids in the fixed order above.

## Expected outputs

- `results/comparison.json` with exact identities, three timings per setting,
  medians, evaluations, throughput, speedups, tensor hashes, and finiteness.
- Four `[16,3,32,32]` model-space sample tensors.
- Four inverse-normalized 4x4 grids.
- One labeled DDPM/DDIM headline comparison figure.
- One runtime-versus-evaluation figure or table.
- Slurm log and run manifest with source/config/checkpoint/system hashes.

## Acceptance criteria

- Unit tests independently verify timestep selection, one-step arithmetic,
  final-step semantics, deterministic repeatability, finiteness, validation,
  and exact model-call counts.
- Local tests, Ruff lint, Ruff formatting, shell syntax, and remote CI pass.
- The Slurm job enforces a clean exact run commit and exact checkpoint hash.
- All four settings produce finite tensors of shape `[16,3,32,32]`.
- Observed model evaluations are exactly `1000`, `100`, `50`, and `25` per
  repeat.
- Each setting's three output hashes match exactly.
- Each DDIM median runtime is below the DDPM-1000 median runtime.
- The fixed-seed grids and headline figure are visually inspected without
  seed substitution.

Visual degradation is an observation, not a structural failure. If a shorter
trajectory is visually poor but mathematically valid, report it honestly.

## Failure and stop conditions

- Stop on source, config, checkpoint, shape, timestep, call-count, or hash
  mismatch.
- Stop on nonfinite model output, clean estimate, or sample.
- Do not run outside Slurm or on an unvalidated GPU type.
- Preserve a failed try; do not overwrite it.
- Do not retrain, extend to 100k, add FID/KID, tune seeds, add more step counts,
  or implement additional algorithms.
- Stop after one valid comparison and proceed to recruiter packaging.

## Compute and storage

- One H100 through Slurm; no login-node compute.
- Reuse the persistent PyTorch 2.13.0+cu130 environment.
- Copy the checkpoint into job-local scratch and verify its hash.
- Keep active work and caches in job-local scratch.
- Stage only JSON, PNG, small sample tensors, and logs to
  `~/data/diffusion-models-from-scratch/exp005-ddim-comparison/try01/`.

## Documentation updates

Update the experiment overview, hypothesis, try metadata/report, Milestone 7
report, README, project specification, state, context, timeline, thinking, and
experiment index only after evidence is validated.

## Unresolved items

Actual runtimes and qualitative differences are unknown and nonblocking. They
must be observed rather than predicted. Distribution-level quality remains
explicitly unevaluated.
