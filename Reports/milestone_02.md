# Milestone 02 Report: Time-Conditioned CIFAR-10 U-Net

## Objective

Implement the frozen primary and smoke U-Net architectures from
`docs/plans/unet.md` without beginning training or reverse sampling.

## Baseline identity

- Repository baseline: commit `6183fef`
- Frozen forward-process files preserved:
  `src/diffusion_models/diffusion/schedules.py` and
  `src/diffusion_models/diffusion/ddpm.py`
- Dataset/model convention: RGB `[B,3,32,32]` tensors in `[-1,1]`

## Implemented

- Fixed sinusoidal timestep embedding and learned timestep MLP.
- Time-conditioned residual blocks with equal/projected skip paths.
- Stride-two convolutional downsampling.
- Nearest-neighbor plus convolutional upsampling.
- Residual single-head spatial self-attention.
- Complete encoder, bottleneck, decoder, and deterministic skip consumption.
- Frozen primary and CPU smoke YAML configurations.
- CPU-only GitHub Actions workflow with no dataset or training step.
- Stable fixed-seed empirical conditional-moment check for `q_sample`.
- Interview-oriented U-Net explanation and exercises.

## Exact architecture bookkeeping

| Configuration | Resolutions | Channels | Attention | Trainable parameters |
|---|---|---|---|---:|
| Primary | 32, 16, 8, 4 | 64, 128, 128, 256 | 16×16 encoder and decoder | 12,852,547 |
| Smoke | 32, 16 | 32, 64 | none | 491,107 |

Both use three input and output channels. Every residual block receives a
learned timestep embedding. The primary model remains below its strict
25,000,000-parameter budget.

## Verification

```text
.venv/bin/python -m pytest -q tests/test_unet.py
  -> 14 passed
.venv/bin/python -m pytest -q tests/test_forward_diffusion.py
  -> 4 passed
.venv/bin/python -m pytest -q
  -> 22 passed
.venv/bin/ruff check .
  -> All checks passed
.venv/bin/ruff format --check .
  -> 25 files already formatted
```

The fixed-seed empirical check observed conditional mean `1.7332053583` versus
`1.7320508076` expected, and variance `0.2491815217` versus `0.25` expected.
Its measured pytest call duration was approximately `0.03` seconds locally;
the analytical schedule, formula, and inversion tests remain the primary
correctness evidence.

A direct CPU verification instantiated both configurations and ran one
`[1,3,32,32]` forward/backward pass at timestep 500. Both outputs were finite,
had shape `[1,3,32,32]`, and produced present, finite `.grad` tensors for every
trainable parameter. This does not mean every gradient was nonzero. On the
first primary-model backward pass, each attention output-projection weight and
bias had nonzero gradients, while the corresponding QKV and normalization
gradients were present and finite but exactly zero because the zero-initialized
output projection blocked upstream signal. Observed local elapsed times were
approximately 0.074 seconds for the primary model and 0.011 seconds for the
smoke model; these are environment specific and not benchmark claims.

## Observations and interpretation

The tests establish the specified shape contracts, initialization behavior,
time conditioning, gradient connectivity, parameter budget, and deterministic
architecture placement. They do not establish that the network can learn,
predict noise accurately, or generate images because no optimization was run.

## Scope confirmation

No training, reverse sampling, checkpointing, EMA, dataset download, GPU job,
or external compute was started. CPU CI is configured but has not yet executed
remotely. The changes remain uncommitted pending the final local gate.

## Next action

Review and commit this milestone. Training and tiny-subset overfit require a
separate authorized plan and implementation request.
