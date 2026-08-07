# DDPM Reverse Process

## Goal and assumptions

The forward chain is fixed:

```math
q(x_t\mid x_{t-1})=\mathcal N(\sqrt{\alpha_t}x_{t-1},\beta_t I),
\qquad
q(x_t\mid x_0)=\mathcal N(\sqrt{\bar\alpha_t}x_0,
(1-\bar\alpha_t)I).
```

Here `alpha_t=1-beta_t`, `bar(alpha)_t=prod_{s=1}^t alpha_s`,
`0<beta_t<1`, and `bar(alpha)_0=1`. Conditioned on both `x_t` and `x_0`,
the previous state has a tractable Gaussian posterior.

## Deriving the posterior

Bayes' rule gives, up to a normalization constant independent of `x_{t-1}`,

```math
q(x_{t-1}\mid x_t,x_0)
\propto q(x_t\mid x_{t-1})q(x_{t-1}\mid x_0).
```

Substitute the two Gaussian densities and retain the quadratic terms in
`x_{t-1}`:

```math
-2\log q =
\frac{\lVert x_t-\sqrt{\alpha_t}x_{t-1}\rVert^2}{\beta_t}
+\frac{\lVert x_{t-1}-\sqrt{\bar\alpha_{t-1}}x_0\rVert^2}
{1-\bar\alpha_{t-1}}+C.
```

The precision multiplying the quadratic term is

```math
\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar\alpha_{t-1}}
=\frac{1-\bar\alpha_t}{\beta_t(1-\bar\alpha_{t-1})}.
```

Its inverse is the posterior variance:

```math
\boxed{
\tilde\beta_t=
\beta_t\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}
}.
```

Multiplying the linear term by this variance gives the posterior mean:

```math
\boxed{
\tilde\mu_t(x_t,x_0)=
\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0
+\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}
{1-\bar\alpha_t}x_t
}.
```

Therefore

```math
q(x_{t-1}\mid x_t,x_0)
=\mathcal N(\tilde\mu_t(x_t,x_0),\tilde\beta_t I).
```

At `t=1`, `bar(alpha)_0=1`, so `tilde(beta)_1=0`, the coefficient
on `x_t` is zero, and the coefficient on `x_0` is one. The last reverse
transition is consequently deterministic.

## Replacing unknown clean data with a model prediction

During generation `x_0` is unknown. The network predicts the epsilon in the
forward marginal:

```math
\epsilon_\theta(x_t,t)\approx\epsilon,
\qquad
x_t=\sqrt{\bar\alpha_t}x_0+sqrt{1-\bar\alpha_t}\epsilon.
```

Solving the same forward equation for clean data gives

```math
\boxed{
\hat x_0(x_t,t)=
\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta(x_t,t)}
{\sqrt{\bar\alpha_t}}
}.
```

The implementation reuses `predict_x_start_from_noise` for this algebra,
optionally clips `hat(x)_0` to model space `[-1,1]`, and substitutes it into
the exact posterior mean. This defines

```math
p_\theta(x_{t-1}\mid x_t)
=\mathcal N(\tilde\mu_t(x_t,\hat x_0),\tilde\beta_t I).
```

For `t>1`, ancestral sampling uses

```math
x_{t-1}=\tilde\mu_t+\sqrt{\tilde\beta_t}z,
\qquad z\sim\mathcal N(0,I).
```

For `t=1`, it returns the mean without drawing or adding `z`.

## Code indexing and loop labels

Mathematical timestep `t` maps to code index `i=t-1`. Code index zero applies
the first beta, and its posterior uses `bar(alpha)_0=1`; no identity entry is
added to the stored schedule.

The reverse loop begins with `x_T` and visits code indices `T-1,...,0`. After
processing code index `i`, the new state has `i` remaining transitions, so a
captured trajectory uses key `i`. Key `T` is reserved for the initial Gaussian.
This makes keys `[T,750,500,250,100,0]` denote
`[x_T,x_750,x_500,x_250,x_100,x_0]` when `T=1000`.

Schedule scalars are gathered per example and reshaped from `[B]` to
`[B,1,1,1]`, so posterior coefficients broadcast across channels and pixels
without constructing full image-sized coefficient tensors.

## What this establishes

Posterior arithmetic, final-step masking, seeded stochastic transitions, and a
complete finite loop establish structural sampling correctness. A trajectory
from randomly initialized weights is intentionally meaningless as an image-
quality evaluation. Quality requires a trained checkpoint and a separately
frozen evaluation protocol.
