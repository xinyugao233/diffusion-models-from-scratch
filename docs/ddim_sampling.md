# Deterministic DDIM sampling

DDIM changes the reverse trajectory, not the training objective. The same
epsilon-prediction U-Net trained for DDPM can therefore be sampled without
retraining.

## From a noise prediction to a clean-image prediction

The DDPM forward marginal is

```text
x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) epsilon.
```

Solving for `x_0` after replacing the unknown noise with the model prediction
gives

```text
x0_hat = (x_t - sqrt(1 - alpha_bar_t) epsilon_theta(x_t, t))
         / sqrt(alpha_bar_t).
```

This repository clamps `x0_hat` to `[-1, 1]`, matching its DDPM sampler.

## The eta=0 transition

Choose a decreasing subsequence of training indices. If `s < t` is the next
selected index, deterministic DDIM uses

```text
x_s = sqrt(alpha_bar_s) x0_hat
      + sqrt(1 - alpha_bar_s) epsilon_theta(x_t, t).
```

For the last transition, define `alpha_bar_-1 = 1`; then the output is exactly
the clipped clean-image estimate. The general DDIM update includes a variance
term controlled by `eta`. Here `eta=0`, so that variance and the added Gaussian
noise are both zero. Given initial `x_T` and a deterministic evaluation-mode denoiser, the whole
trajectory is deterministic within the stated software/hardware environment.

## Timestep selection and compute

`ddim_timesteps` rounds an evenly spaced grid over code indices `[0, T-1]`,
preserves both endpoints, checks uniqueness, and reverses it for sampling. A
trajectory with `S` selected indices calls the denoiser exactly `S` times.
This makes denoiser evaluations an explicit compute measure: 100, 50, and 25
DDIM steps use 10%, 5%, and 2.5% as many evaluations as DDPM-1000.

Wall-clock speedup need not exactly equal the evaluation-count ratio because
checkpoint loading, Python overhead, synchronization, and GPU utilization have
different costs. The Milestone 7 experiment therefore records synchronized
end-to-end trajectory times separately from model-call counts.

## What the comparison can establish

A fixed-seed grid can show a checkpoint-specific quality-compute tradeoff and
make sampling behavior easy to inspect. It cannot establish distribution-level
sample quality. Claims about FID, KID, diversity, or general sampler superiority
require a larger preregistered evaluation.

## Reference

Jiaming Song, Chenlin Meng, and Stefano Ermon, *Denoising Diffusion Implicit
Models*, arXiv:2010.02502 (2020).

## Captured state indices and dtype contract (maintenance correction)

`SamplingResult.trajectory` uses mathematical **state** indices for DDPM and
DDIM: initial noise is T; a destination at code index s is state s+1;
the final clean output is state 0. For T=10 and four DDIM calls, code
indices (9,6,3,0) visit states **10,7,4,1,0**. Requests for unvisited states
fail and list the allowed states. This corrects the previous source-index
labels (10,9,6,3,0), which mislabeled destination tensors. Final samples,
model-call count and reverse RNG use are unchanged by capture bookkeeping.
Historical benchmarks are not regenerated.

The reusable DDPM/DDIM step/loop APIs temporarily select evaluation mode and
restore every module's previous mode, including on exceptions. `no_grad` alone
does not disable dropout. Portable inference loads an evaluation-mode model.
CPU float32 and float64 are covered by tests; model parameters, initial tensors
and the sampling dtype/device must agree. The U-Net casts sinusoidal features
to its projection dtype. Reduced-precision loop requests are rejected; no
reduced-precision or cross-device bitwise equivalence is advertised.
