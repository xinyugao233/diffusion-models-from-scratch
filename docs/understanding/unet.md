# Understanding the Time-Conditioned U-Net

## Purpose

The U-Net receives a noised image `x_t` and its diffusion timestep `t`. It
predicts the Gaussian noise that was added to the clean image:

```math
\epsilon_\theta(x_t,t)\approx\epsilon,
\qquad
x_t=\sqrt{\bar\alpha_t}x_0+sqrt{1-\bar\alpha_t}\epsilon.
```

It predicts noise rather than a finished image. A later training milestone will
minimize mean squared error between this output and the sampled noise.

## Why the output has three channels

CIFAR-10 inputs are RGB tensors `[B,3,32,32]`. The target noise has exactly the
same shape as the input, so the final convolution must produce three channels:

```text
x_t              [B, 3, 32, 32]
predicted noise  [B, 3, 32, 32]
```

## Sinusoidal timestep embeddings

A raw timestep is one scalar per image and cannot directly describe noise level
to every convolutional block. `SinusoidalTimeEmbedding` evaluates sine and
cosine functions at geometrically spaced frequencies, converting `[B]` into
`[B,D]`. Nearby and distant timesteps receive distinct, deterministic feature
patterns. A two-layer MLP then learns how to combine these fixed features.

The primary model maps:

```text
t [B] -> sinusoidal features [B,64] -> learned embedding [B,256]
```

## How a residual block uses time

For feature tensor `h` and learned timestep embedding `e_t`, a block performs:

```text
h
 -> GroupNorm -> SiLU -> 3x3 convolution
 -> add Linear(SiLU(e_t))[:, :, None, None]
 -> GroupNorm -> SiLU -> dropout -> 3x3 convolution
 -> add the identity or a 1x1 projected skip
```

The `[B,C]` time projection is reshaped to `[B,C,1,1]`, so broadcasting adds a
different timestep-dependent channel bias to every example. Every residual
block—including both bottleneck blocks—receives this embedding.

If input and output channels differ, a learned 1×1 convolution makes the
residual path compatible. Spatial resolution is unchanged inside the block.

## Encoder and decoder shapes

The frozen primary architecture has two residual blocks per level:

| Stage | Resolution | Channels | Stored/consumed skip |
|---|---:|---:|---|
| Input convolution | 32×32 | 64 | no |
| Encoder level 0 | 32×32 | 64 | store |
| Encoder level 1 | 16×16 | 128 | store |
| Encoder level 2 | 8×8 | 128 | store |
| Encoder level 3 | 4×4 | 256 | store |
| Bottleneck | 4×4 | 256 | no |
| Decoder level 3 | 4×4 | 256 | consume level 3 |
| Decoder level 2 | 8×8 | 128 | consume level 2 |
| Decoder level 1 | 16×16 | 128 | consume level 1 |
| Decoder level 0 | 32×32 | 64 | consume level 0 |
| Output convolution | 32×32 | 3 | no |

Stride-two 3×3 convolutions perform downsampling. Nearest-neighbor resizing
followed by a 3×3 convolution performs upsampling.

## Skip-connection data flow

The encoder appends exactly one tensor after each level. The decoder pops these
tensors in last-in, first-out order. At each resolution, it verifies batch and
spatial compatibility, concatenates the decoder and skip tensors along the
channel dimension, and reduces them through a residual block. A final assertion
ensures no skip tensor remains unused.

Skip connections return high-resolution information that would otherwise have
to pass through the narrow bottleneck. They also provide short gradient paths.

## Why attention is at 16×16

Convolutions are local. Self-attention lets any of the 256 positions at 16×16
interact directly, helping coordinate nonlocal structure while keeping the
quadratic attention matrix manageable. The primary configuration uses one
attention block at 16×16 in both encoder and decoder. It uses no attention at
32×32, 8×8, 4×4, or in the bottleneck. The smoke configuration disables it.

The attention output projection starts at zero, so the residual attention block
initially behaves exactly as the identity. It can learn a nonzero correction.
On the first backward pass, the output projection receives nonzero gradients,
while the earlier QKV and normalization parameters have present and finite but
zero-valued gradients because the zero projection blocks upstream signal. After
an optimizer changes the output projection, later passes can send nonzero
gradients upstream. “Receives a gradient” therefore means `.grad` exists; it
does not mean every first-step gradient is nonzero.

## Component contracts

| Component | Input | Output |
|---|---|---|
| `SinusoidalTimeEmbedding(D)` | timestep `[B]` | features `[B,D]` |
| `TimestepEmbedding` | timestep `[B]` | learned features `[B,time_dim]` |
| `ResidualBlock` | image `[B,Cin,H,W]`, time `[B,time_dim]` | `[B,Cout,H,W]` |
| `Downsample` | `[B,Cin,H,W]` | `[B,Cout,H/2,W/2]` |
| `Upsample` | `[B,Cin,H,W]` | `[B,Cout,2H,2W]` |
| `SelfAttention2d` | `[B,C,H,W]` | `[B,C,H,W]` |
| `CIFAR10UNet` | image `[B,3,32,32]`, timestep `[B]` | noise `[B,3,32,32]` |

All paired tensors must already share a device. The model never moves inputs.

## Important functions

- `primary_unet_config`: frozen portfolio-scale architecture.
- `smoke_unet_config`: small architecture for fast CPU integration checks.
- `CIFAR10UNet.forward`: validates inputs, creates time features, records
  encoder skips, processes the bottleneck, and consumes skips in reverse.
- `count_trainable_parameters`: exposes the exact optimization parameter count.

## Common implementation bugs

- Treating `t` as one scalar for the whole batch instead of `[B]`.
- Forgetting to reshape the time projection to `[B,C,1,1]`.
- Concatenating skips in encoder order instead of reverse order.
- Concatenating tensors before their spatial resolutions match.
- Forgetting a 1×1 residual projection when channel counts change.
- Applying attention at every resolution and unexpectedly exhausting memory.
- Returning three image channels from an internal level rather than the final
  full-resolution feature tensor.
- Moving inputs to a hard-coded device inside `forward`.
- Assuming GroupNorm always supports 32 groups when channels are not divisible.

## Interview questions and concise answers

1. **What does the network predict?** The Gaussian noise used to construct
   `x_t` from `x_0`.
2. **Why is timestep conditioning necessary?** The correct denoising operation
   changes with the signal-to-noise ratio.
3. **How does `[B]` become `[B,256]`?** Fixed sinusoidal features followed by a
   learned two-layer MLP.
4. **How does time enter a residual block?** A learned channel projection is
   broadcast and added after the first convolution.
5. **Why use multiple resolutions?** Low resolutions efficiently build broad
   context; high resolutions preserve detail.
6. **Why use skip connections?** They restore encoder spatial detail and shorten
   gradient paths.
7. **Why does the output have three channels?** The target noise matches RGB
   input shape.
8. **Why attention only at 16×16?** It adds nonlocal interaction at a manageable
   quadratic cost.
9. **Why nearest-neighbor upsampling plus convolution?** It is simple and avoids
   common checkerboard artifacts from transposed convolutions.
10. **What is learned in the forward process?** Nothing; the schedule and
    `q_sample` are fixed, while only U-Net parameters will be optimized.

## Modification exercises

### Change the base width

Construct `UNetConfig(base_channels=48, time_embedding_dim=192)` and instantiate
the model. Print `trainable_parameter_count`, run a smoke forward/backward pass,
and confirm the GroupNorm divisor rule remains valid. Do not overwrite the
frozen primary YAML.

### Disable attention

Use `dataclasses.replace(primary_unet_config(), attention_resolutions=())`, run
the same input/output and backward checks, and compare its parameter count with
the frozen primary model. The output contract must remain `[B,3,32,32]`.
