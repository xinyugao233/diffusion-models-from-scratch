# Time-Conditioned CIFAR-10 U-Net Plan

## Objective

Implement a readable PyTorch U-Net that predicts the Gaussian noise in a
noised CIFAR-10 tensor. This is an engineering milestone; it does not train or
evaluate a denoiser.

## Existing evidence and invariants

Milestone-one commit `6183fef` validates the linear schedule, coefficient
extraction, `q_sample`, exact-noise inversion, and `[-1,1]` image convention.
Those implementations and tests are frozen. Code timestep `0` remains the first
noising step rather than an identity step.

## Public interface

```python
predicted_noise = model(x_t, timesteps)
```

- `x_t`: floating tensor `[B,3,32,32]`
- `timesteps`: `torch.long` tensor `[B]`, on the same device
- output: floating tensor `[B,3,32,32]`, on the input device

The model never moves caller tensors between devices.

## Frozen primary architecture

```yaml
image_size: 32
in_channels: 3
out_channels: 3
base_channels: 64
channel_multipliers: [1, 2, 2, 4]
residual_blocks_per_level: 2
normalization: group_norm
group_norm_groups: 32
activation: silu
dropout: 0.1
attention_resolutions: [16]
time_embedding_dim: 256
downsampling: stride_2_convolution
upsampling: nearest_neighbor_plus_convolution
parameter_budget_max: 25000000
```

Expected encoder resolutions and channels are `(32,64)`, `(16,128)`,
`(8,128)`, and `(4,256)`. One skip tensor is retained after each encoder level
and consumed by the matching decoder level in reverse order. Each level has two
time-conditioned residual blocks. Self-attention appears only after the 16×16
residual blocks in the encoder and decoder. The bottleneck uses residual blocks
without attention so the stated attention-resolution invariant remains exact.

For GroupNorm, choose the largest divisor of the channel count that does not
exceed the configured maximum of 32. This makes the rule valid for projected or
future channel widths without silently changing normalization type.

## Frozen CPU smoke architecture

```yaml
image_size: 32
in_channels: 3
out_channels: 3
base_channels: 32
channel_multipliers: [1, 2]
residual_blocks_per_level: 1
normalization: group_norm
group_norm_groups: 32
activation: silu
dropout: 0.0
attention_resolutions: []
time_embedding_dim: 128
downsampling: stride_2_convolution
upsampling: nearest_neighbor_plus_convolution
```

## Components and expected files

- `src/diffusion_models/models/unet.py`: configuration, sinusoidal embedding,
  timestep MLP, residual block, downsample, upsample, self-attention, U-Net, and
  parameter counting.
- `src/diffusion_models/models/__init__.py`: public model exports.
- `configs/unet_primary.yaml` and `configs/unet_smoke.yaml`: frozen readable
  configurations.
- `tests/test_unet.py`: focused CPU component and integration tests.
- `tests/test_forward_diffusion.py`: one fixed-seed empirical distribution test.
- `.github/workflows/ci.yml`: lint, format, and CPU tests without dataset access.
- `docs/understanding/unet.md`: interview-oriented explanation.

## Verification

Tests must cover embedding determinism and validation; equal/projected residual
paths and timestep influence; attention shape, residual initialization, and
backpropagation; full-model batch and shape contracts; finite outputs and
gradients; gradient reachability; parameter budget; and fast smoke execution.
The empirical forward check uses a scalar float64 setup and a fixed seed with
tolerances justified from sampling error.

## Stop and failure conditions

- Stop if the primary model reaches or exceeds 25,000,000 trainable parameters.
- Stop if encoder and decoder resolutions or skip shapes cannot be matched
  exactly under the frozen architecture.
- Do not weaken tests, remove 16×16 attention, or change the architecture to
  make validation pass without recording and authorizing a plan revision.
- Do not begin training, sampling, checkpointing, EMA, GPU work, or external
  compute.

## Acceptance criteria

All specified component and full-model tests pass on CPU; Ruff lint and format
checks pass; primary and smoke parameter counts are reported; a primary and
smoke forward/backward pass completes with finite outputs and gradients; CI
performs the same non-data-dependent checks; documentation explains the exact
tensor flow; and milestone-one behavior remains unchanged.
