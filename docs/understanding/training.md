# Understanding Epsilon-Prediction Training

## The objective

For each clean image `x_start = x_0`, sample a code timestep `t` and Gaussian
noise `epsilon`, then form

```math
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon.
```

The time-conditioned U-Net predicts `epsilon_theta(x_t,t)`. Training minimizes

```math
L=\mathbb{E}_{x_0,t,\epsilon}
\left[\lVert\epsilon-\epsilon_\theta(x_t,t)\rVert_2^2\right].
```

In code, PyTorch's mean-squared error averages the squared residual over the
batch, channel, height, and width dimensions.

## Why predict noise

The sampled noise is an exactly known supervised target at every timestep. It
also has a stable standard-normal scale, unlike a target whose numerical scale
changes with the image or timestep. Once the model estimates noise, the DDPM
equations can algebraically convert that estimate into an estimate of `x_0` or
the reverse-process mean. Noise prediction is a parameterization choice: the
network learns a denoising direction rather than directly emitting a display
image.

## Why every image gets a random timestep

One network must denoise all noise levels. Sampling `t` independently for every
image turns the expected objective over timesteps into an ordinary minibatch
estimate and exposes a single update to several signal-to-noise ratios. Using
one timestep for an entire training run would train only one conditional
problem; omitting `t` from the model would make different noise levels
ambiguous.

Code timestep `0` is the first DDPM step, not an unnoised identity step. In this
project `t` is uniformly sampled from integer indices `[0,999]`.

## Why the same noise has two roles

The target must be the exact latent random variable that produced the input.
If `epsilon_a` constructs `x_t` but the loss targets an independent
`epsilon_b`, the input contains no information identifying `epsilon_b`. The
regression target then becomes inconsistent and optimization cannot learn the
intended denoising function.

The central wiring therefore stays visible:

```python
timesteps = torch.randint(0, schedule.num_steps, (x_start.shape[0],))
noise = torch.randn_like(x_start)
x_t = q_sample(x_start, timesteps, schedule, noise=noise)
predicted_noise = model(x_t, timesteps)
loss = torch.nn.functional.mse_loss(predicted_noise, noise)
```

## Tensor shapes through one update

| Tensor | Shape | Meaning |
|---|---|---|
| `x_start` | `[B,3,32,32]` | normalized CIFAR-10 images |
| `timesteps` | `[B]` | one integer schedule index per image |
| `noise` | `[B,3,32,32]` | independently sampled standard Gaussian target |
| schedule coefficients | `[B,1,1,1]` | extracted and broadcast per image |
| `x_t` | `[B,3,32,32]` | closed-form noised inputs |
| `predicted_noise` | `[B,3,32,32]` | U-Net output |
| residual | `[B,3,32,32]` | `predicted_noise - noise` |
| `loss` | `[]` | scalar mean of squared residuals |

All tensors and the schedule must share a device. Images and noise must share a
floating dtype; timesteps use `torch.long`.

## What backward computes

The call is `loss.backward()`, not `model.backward()`. Autograd follows the
recorded computation graph from the scalar loss through MSE, the U-Net, and all
trainable operations. For each parameter `theta`, it accumulates
`partial loss / partial theta` in `theta.grad`. Backward computes gradients; it
does not change parameter values.

A gradient tensor can exist while every entry is zero. In the zero-initialized
attention block, the first backward pass gives the output projection a nonzero
gradient but blocks upstream QKV gradients. After the projection changes, a
later pass can produce nonzero QKV gradients.

## What optimizer.step changes

`optimizer.step()` reads the current parameter gradients and optimizer state,
then updates the parameter tensors in place. AdamW maintains moving averages of
the gradient and squared gradient and applies decoupled weight decay. In this
milestone weight decay is zero, so only the adaptive gradient update changes
the U-Net.

## What zero_grad does

PyTorch accumulates gradients by default. `optimizer.zero_grad(set_to_none=True)`
clears references before the next backward pass so an update uses only the
current training step. Forgetting it unintentionally sums gradients across
steps and changes the effective optimization rule.

## What gradient clipping does

Global L2 clipping calculates one norm over every parameter gradient. When that
norm exceeds a threshold `c`, all gradients are scaled by the same factor so
the resulting norm is at most `c`. This limits unusually large updates but does
not fix a nonfinite loss or gradient; the implementation rejects NaN or
infinity before clipping.

## What the fixed-16 overfit proves

A substantial early-to-late loss reduction shows that the data transform,
schedule, per-image timestep sampling, noise construction, U-Net call, MSE,
backpropagation, gradient handling, and optimizer updates form a learnable
pipeline on the selected examples. It is a high-value integration test because
many wiring errors prevent even tiny-data fitting.

## What it does not prove

The run does not evaluate held-out images, full-dataset convergence, sampling,
reverse-process equations, perceptual quality, diversity, likelihood, FID,
checkpoint resumption, or training stability at scale. Memorizing or learning
16 examples is intentionally easier than training a useful generative model.

## Common training-wiring bugs

- Sampling one noise tensor for `q_sample` and a different tensor for the loss.
- Sampling one scalar timestep instead of one timestep per image.
- Using mathematical one-based timesteps directly as zero-based tensor indices.
- Forgetting to normalize CIFAR-10 from `[0,1]` to `[-1,1]`.
- Applying random augmentation to a supposedly frozen overfit subset.
- Passing floating timesteps when the schedule extractor expects `torch.long`.
- Keeping the schedule on CPU while model inputs are on another device.
- Broadcasting coefficients over the wrong dimension.
- Calling `optimizer.step()` before `loss.backward()`.
- Forgetting `zero_grad` and unintentionally accumulating gradients.
- Clipping or stepping after a NaN rather than stopping immediately.
- Comparing only the first and last stochastic losses instead of window means.
- Saving only a plot while losing the raw scalar log and exact configuration.

## Interview questions and concise answers

1. **What is the epsilon-prediction target?** The exact Gaussian tensor used to
   construct each `x_t`.
2. **Why is MSE appropriate here?** The DDPM parameterization reduces the
   variational training term to weighted Gaussian-noise regression; the common
   simplified objective uses unweighted MSE.
3. **Why condition on `t`?** The optimal denoising function changes with the
   timestep's signal-to-noise ratio.
4. **Why sample `t` uniformly?** It gives a simple unbiased Monte Carlo estimate
   of the equally weighted objective over discrete timesteps.
5. **Can the target noise be resampled after making `x_t`?** No. It must be the
   same realization or the supervised pair is wrong.
6. **Does backward update weights?** No. It computes and accumulates gradients;
   the optimizer performs the update.
7. **Why clear gradients each step?** Gradients accumulate by default, while
   this training rule intends one fresh minibatch estimate per update.
8. **What does a global gradient norm summarize?** The Euclidean magnitude of
   all trainable parameter gradients treated as one vector.
9. **Why use early and late windows?** Fresh noise and timesteps make individual
   losses fluctuate, while window means better reveal the optimization trend.
10. **Why run a fixed synthetic gate first?** It isolates optimizer and target
    wiring from dataset, transform, and stochastic-resampling concerns.
11. **What does tiny-subset overfitting diagnose?** Whether the end-to-end
    training path has enough correct signal and capacity to learn repeated data.
12. **What is the next missing capability after this milestone?** Reliable
    training-state persistence and reverse sampling, neither of which this loss
    experiment tests.
