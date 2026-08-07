# Understanding DDPM Sampling

## Why the posterior is tractable

Every forward transition is linear with additive Gaussian noise, and the
closed-form distribution of `x_{t-1}` given `x_0` is also Gaussian. Multiplying
the likelihood `q(x_t | x_{t-1})` by the prior
`q(x_{t-1} | x_0)` therefore produces another Gaussian in `x_{t-1}`.
Completing the square gives its exact mean and variance; no numerical inference
is needed.

## How epsilon prediction becomes an estimate of clean data

Training constructs

```math
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon.
```

The U-Net predicts the epsilon term. Replacing epsilon with the network output
and solving this equation for `x_0` produces `hat(x)_0`. The sampler optionally
clips that estimate to the model-space bounds and places it into the exact
posterior mean. This is why the same epsilon-prediction model used by the
training loss can parameterize the reverse transition.

## Why no noise is added on the final step

The first mathematical reverse posterior uses `bar(alpha)_0=1`, making
`tilde(beta)_1=0`. Its Gaussian collapses to a point mass. In code this is the
zero-based timestep `0`, so `p_sample` returns the mean directly and does not
consume the supplied generator. Adding noise there would contradict the
derived posterior and corrupt the final sample.

## Forward noise versus reverse noise

Forward noise is the epsilon used to create a training input `x_t` directly
from clean `x_0`. It defines the regression target for epsilon prediction.

Reverse noise is a fresh `z` used after the model has predicted epsilon and the
posterior mean has been constructed. It samples uncertainty in
`p_theta(x_{t-1} | x_t)`. It is independent at each nonfinal reverse step and
is never added at the final step. The two noises have different roles even
though both are standard Gaussian tensors.

## Code structure worth explaining in an interview

- `q_posterior_mean_variance` contains only exact schedule algebra.
- `p_mean_variance` calls the epsilon model, reuses the forward inversion, and
  substitutes `hat(x)_0` into the posterior.
- `p_sample` adds the correctly scaled reverse noise with an explicit
  timestep-zero mask.
- `p_sample_loop` starts at Gaussian `x_T`, visits indices `T-1` through `0`,
  checks every state is finite, and can capture debugging states.
- EMA is deliberately outside this stack: copy EMA weights into a model and
  pass that model through the same interface. This avoids duplicated sampler
  code paths and makes the weight source explicit.

## What the smoke figure means

The six-column `EXP003/try01` trajectory verifies state ordering, tensor range
handling at the display boundary, and successful execution of all 1,000 calls.
It remains noise-like because the model is randomly initialized. That visual is
a debugging artifact, not a failed quality result and not evidence that the
model learned CIFAR-10.
