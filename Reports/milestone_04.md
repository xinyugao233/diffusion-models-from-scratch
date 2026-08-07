# Milestone 04 Report: EMA and Exact Checkpoint Resume

## Objective

Provide explicit exponential moving average state and a structured training
checkpoint that resumes the deterministic CPU epsilon-prediction trajectory
without changing any later update or random draw.

## Baseline and publishing gate

- Baseline commit: `45c17b8a743b56b0233794dfbc25b8709863473c`
- Branch and remote: `main`, synchronized with `origin/main` before Milestone 4
- Milestone 3 GitHub Actions: run `31151451025`, SUCCESS, 49 seconds
- Local baseline: clean before `EXP002` creation
- Executed Milestone 4 code: uncommitted, with exact hashes recorded below

## Implemented behavior

- Detached EMA initialization from every named model parameter and buffer.
- Explicit post-optimizer update
  `ema = decay * ema + (1 - decay) * model`.
- Exact-copy buffer policy for floating and integer buffers.
- Strict EMA state save/load and future `copy_to(model)` evaluation interface.
- Versioned state-dictionary checkpoint with model, EMA, AdamW, global step,
  configuration, Python RNG, torch CPU RNG, optional named generator states,
  optional explicitly used CUDA RNG states, and optional NumPy RNG state.
- Strict missing/corrupt/incompatible/configuration-mismatch errors.
- Refusal to overwrite an existing checkpoint.
- Exact nested state comparison across compatible mapping subclasses.

No U-Net, forward-diffusion, or reverse-sampling code was changed.

## Verification

```text
.venv/bin/python -m pytest -q
  -> 41 passed
.venv/bin/ruff check .
  -> All checks passed
.venv/bin/ruff format --check .
  -> 50 files already formatted
git diff --check
  -> passed with no output
```

Normal tests use synthetic CPU data and contain only a short trajectory check;
they do not download CIFAR-10.

## Try 01: preserved validation failure

The first frozen 100-vs-50+50 run completed but returned `passed=false` because
the validation helper required identical container classes before comparing
values. PyTorch's model state was an `OrderedDict`; the cloned uninterrupted
snapshot was a plain `dict`. `model_state_exact` was therefore false without
visiting any model tensor.

Every other comparison was exact, including EMA, optimizer, complete loss
trajectory, final loss, global step, EMA update count, next Python draw, next
timesteps, next noise, and next loss. This evidence did not justify relabeling
try01 as passed. Its checkpoint and summary are preserved under `try01`.

## Try 02: exact deterministic resume

Try02 changed only mapping-subclass comparison and output paths. Seeds, model,
synthetic images, optimizer, schedule, EMA, gradient clipping, device, dtype,
100 total steps, 50-step interruption, and exact acceptance rules remained
unchanged.

```text
Command: .venv/bin/python scripts/run_resume_validation.py \
         --config configs/resume_validation_try02.json
Model: smoke U-Net, 491,107 parameters
Batch: 4 fixed synthetic images
Schedule: linear T=1000, beta 1e-4 to 2e-2
Optimizer: AdamW, lr 1e-3, weight decay 0
EMA decay: 0.99
Device/dtype: one Apple arm64 CPU thread / float32
Path A: 100 uninterrupted steps, 4.5929 seconds
Path B: 50 + checkpoint/reconstruct/restore + 50 steps, 4.6884 seconds
Images processed per path: 400 (0.4 kimg, 0.0004 mimg)
Final loss: 0.16175299882888794
Next loss: 0.41868576407432556
Next timesteps: [12, 899, 157, 802]
Result: PASS
```

All exact comparisons passed:

- every model tensor;
- every EMA tensor and update count (`100`);
- complete nested AdamW state;
- full 100-loss trajectory and final loss;
- global step;
- next Python draw;
- next timestep tensor;
- next Gaussian-noise tensor;
- next loss.

The negative control reconstructed and loaded the checkpoint without restoring
RNG. Its next timestep/noise draw differed, confirming that the validation
actually depends on RNG restoration.

## Checkpoint audit

The try02 intermediate checkpoint is 7,983,729 bytes with SHA-256
`3d4ff5184457c4002b2f7a8bb192360e9391989bbe9f523c3c8cc30f474012c0`.
Independent inspection found:

- checkpoint schema version `1` and the exact eight required top-level keys;
- global step `50`;
- 78 model tensors;
- 78 EMA parameters and EMA update count `50`;
- 78 populated optimizer-state entries;
- complete frozen JSON configuration;
- Python and torch CPU RNG state;
- no NumPy state because NumPy randomness was not used;
- no CUDA state because CUDA was not used.

The try02 result summary SHA-256 is
`9cf70d48911aabc2a64f862978d472b17542aae688aae3d6f0210d5055c0b2dd`.

## Scientific conclusion

The evidence supports this limited claim:

> On the frozen deterministic CPU configuration and software stack, training
> continuously for 100 steps is bitwise identical to training 50 steps,
> saving/reconstructing/restoring the complete checkpoint, and training the
> remaining 50 steps.

It does not establish exact resume across PyTorch versions, operating systems,
CPU architectures, GPUs, distributed execution, mixed precision, data-loader
workers, or full CIFAR-10 training. It also does not establish sampling or
generation quality.

A PDF report was not generated because this is a state-correctness milestone
with no figure, checkpoint evaluation sample, or presentation request. This
Markdown report, structured summaries, tests, and checkpoint are canonical.

## Evidence paths

- Plan: `docs/plans/ema_and_checkpoints.md`
- Understanding: `docs/understanding/ema_and_checkpoints.md`
- Try01 report: `Experiments/exp002-deterministic-checkpoint-resume/try01/report.md`
- Try02 report: `Experiments/exp002-deterministic-checkpoint-resume/try02/report.md`
- Try02 checkpoint: `Experiments/exp002-deterministic-checkpoint-resume/try02/checkpoints/resume_step_0050.pt`
- Try02 summary: `Experiments/exp002-deterministic-checkpoint-resume/try02/results/resume_summary.json`
