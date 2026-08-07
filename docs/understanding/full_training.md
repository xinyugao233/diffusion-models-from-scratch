# Understanding Full DDPM Training

## One complete optimization step

Training samples a real clean image batch `x_0`, one random timestep `t` per
image, and one Gaussian epsilon tensor. The validated forward marginal creates
`x_t`, and the U-Net predicts the epsilon that produced it. Mean squared error
is differentiated only through the trainable U-Net.

The required order is:

```text
optimizer.zero_grad(set_to_none=True)
loss = mse(model(x_t, t), epsilon)
loss.backward()
check and clip gradients
optimizer.step()
ema.update(model)
```

### `zero_grad`

PyTorch accumulates gradients by default. Clearing them prevents the previous
batch's gradient from being added to the current batch. `set_to_none=True`
also avoids unnecessary zero-filled tensors and makes an absent gradient easier
to detect.

### `backward`

Backpropagation computes the current loss gradient with respect to every
trainable parameter. The production path rejects missing or nonfinite gradients
before any optimizer update.

### `optimizer.step`

AdamW uses the current gradients and its momentum state to update the raw U-Net
parameters. These are the only parameters registered with the optimizer.

### Why EMA follows the optimizer

EMA is intended to smooth the sequence of updated models:

```math
\theta_{ema,s}=\gamma\theta_{ema,s-1}+(1-\gamma)\theta_s.
```

Updating EMA before `optimizer.step()` would average the previous raw model
again and create a one-step lag inconsistent with this definition. EMA receives
no gradients and no optimizer step.

## Raw training model versus EMA model

The raw model changes directly at every AdamW step and is used to compute the
next training loss. The EMA object is a detached, smoothed state used for
evaluation. At a checkpoint, EMA parameters are copied into a normal U-Net and
that U-Net is passed to the same sampler as any other model. There is no
EMA-specific sampling algorithm.

## Training versus sampling

Training performs one randomly selected denoising problem per batch:

```text
random real x_0 + random t + random epsilon
    -> one x_t
    -> one U-Net call
    -> one epsilon loss
```

Sampling starts from Gaussian `x_T` and repeatedly calls the U-Net at every
reverse timestep:

```text
random x_T -> 1,000 reverse model calls -> generated x_0
```

Training is stochastic supervision across noise levels. Sampling is a complete
iterative generative trajectory.

## Why timesteps are sampled randomly

The simplified DDPM objective is an expectation over data, timesteps, and
Gaussian noise. Sampling timesteps uniformly gives an unbiased Monte Carlo
estimate of that objective while exposing the model to early, middle, and late
noise levels throughout training. Running an entire 1,000-step chain for every
training image would be unnecessary and dramatically more expensive.

## What decreasing epsilon loss means

A decreasing loss means the model is getting better, on average under the
training sampling distribution, at predicting the injected epsilon. It is
evidence of optimization and denoising learnability.

It does not by itself prove that full reverse trajectories are correct,
samples are recognizable, likelihood is good, or the model generalizes.
Sampling compounds errors across 1,000 calls, so checkpoint samples are an
independent end-to-end check.

## Why fixed sampling seeds matter

Different initial Gaussian tensors can yield very different images. Reusing
the exact same 16 initial tensors and reverse-noise stream at 10k, 25k, and 50k
holds sampling randomness fixed. Visual differences can then be attributed to
checkpoint state rather than convenient seed selection.

## Decreasing loss but poor samples: common causes

- mismatch between the training prediction target and sampler parameterization;
- incorrect timestep indexing or reverse coefficients;
- evaluating raw weights when EMA was intended;
- insufficient training despite a locally decreasing loss;
- overaggressive clipping or optimization instability hidden by averages;
- wrong data normalization or visualization inversion;
- train/eval mode mistakes, especially dropout;
- corrupted or mismatched checkpoint configuration;
- changing random seeds across checkpoint figures;
- architecture or schedule capacity limitations.

Milestones 1–5 eliminate several implementation causes, but not insufficient
training or baseline limitations.

## Interview questions and concise answers

1. **Why predict epsilon instead of clean pixels directly?**
   The DDPM reparameterization makes the injected epsilon an exactly known
   target at every noise level, yielding a simple, stable MSE objective.

2. **Why does `zero_grad` happen every step?**
   Gradients accumulate in PyTorch; clearing them isolates the current batch's
   Monte Carlo estimate.

3. **What does `backward` compute?**
   It applies the chain rule to compute the loss gradient for all trainable
   U-Net parameters.

4. **Why clip gradients?**
   Global norm clipping limits unusually large updates while preserving the
   direction when the norm exceeds the threshold.

5. **Why update EMA after AdamW?**
   EMA should incorporate the newly updated model `theta_s`, not re-average the
   preceding `theta_{s-1}`.

6. **Does EMA train another network?**
   No. It maintains detached weighted averages and receives neither gradients
   nor optimizer state.

7. **Why use the raw model for training and EMA for samples?**
   The raw model must follow every optimizer update; EMA smooths noisy parameter
   motion and often provides more stable evaluation weights.

8. **Why is one training step much cheaper than one sample?**
   A training step uses one random timestep and one U-Net call, whereas an
   ancestral sample uses 1,000 sequential U-Net calls.

9. **How does resume preserve data ordering?**
   Each global step maps to a deterministic epoch permutation and batch offset,
   so the checkpoint step reconstructs the next batch directly.

10. **Why save optimizer and RNG state?**
    AdamW momentum and future stochastic draws influence every later update;
    weights alone cannot reconstruct the trajectory.

11. **What does a falling DDPM loss fail to prove?**
    It does not prove sample recognizability, sampler correctness, competitive
    quality, or generalization.

12. **Why use identical seeds across checkpoint grids?**
    It turns the figure into a controlled longitudinal comparison rather than a
    comparison of unrelated random outcomes.

13. **Why train by steps instead of epochs?**
    Steps align directly with optimizer state, checkpoint cadence, processed
    images, and cross-run comparisons.

14. **Why start with 20 and 500 steps?**
    They cheaply expose memory, stability, checkpoint, resume, and sampling
    wiring failures before committing to tens of thousands of updates.

