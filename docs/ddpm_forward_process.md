# DDPM Forward Process

## Definition and assumptions

Let `x_0` be a clean data sample. A denoising diffusion probabilistic model
defines a fixed Markov chain that gradually adds Gaussian noise:

```math
q(x_{1:T}\mid x_0)=\prod_{t=1}^{T}q(x_t\mid x_{t-1}),
```

with transition

```math
q(x_t\mid x_{t-1})
=\mathcal N\!\left(x_t;\sqrt{1-\beta_t}\,x_{t-1},\beta_t I\right).
```

The variance schedule satisfies `0 < beta_t < 1`. Define

```math
\alpha_t=1-\beta_t,
\qquad
\bar\alpha_t=\prod_{s=1}^{t}\alpha_s.
```

Equivalently, each transition can be sampled by reparameterization:

```math
x_t=\sqrt{\alpha_t}\,x_{t-1}+\sqrt{1-\alpha_t}\,\epsilon_t,
\qquad \epsilon_t\sim\mathcal N(0,I),
```

where the `epsilon_t` are independent across timesteps and independent of
`x_0`.

## Deriving the closed-form marginal

Apply the transition twice:

```math
\begin{aligned}
x_t
&=\sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_t}\epsilon_t\\
&=\sqrt{\alpha_t\alpha_{t-1}}x_{t-2}
  +\sqrt{\alpha_t(1-\alpha_{t-1})}\epsilon_{t-1}
  +\sqrt{1-\alpha_t}\epsilon_t.
\end{aligned}
```

The two noise terms are independent, zero-mean Gaussians. Their covariance is

```math
\left[\alpha_t(1-\alpha_{t-1})+(1-\alpha_t)\right]I
=(1-\alpha_t\alpha_{t-1})I.
```

Repeating this substitution to `x_0` makes the signal coefficient

```math
\sqrt{\alpha_t\alpha_{t-1}\cdots\alpha_1}=\sqrt{\bar\alpha_t}.
```

The accumulated noise remains Gaussian because it is a linear combination of
independent Gaussians. Its covariance telescopes to

```math
\begin{aligned}
&\sum_{s=1}^{t}(1-\alpha_s)\prod_{j=s+1}^{t}\alpha_j\,I\\
&=\left(1-\prod_{s=1}^{t}\alpha_s\right)I
=(1-\bar\alpha_t)I.
\end{aligned}
```

Therefore the marginal at any timestep is

```math
\boxed{
q(x_t\mid x_0)
=\mathcal N\!\left(x_t;\sqrt{\bar\alpha_t}\,x_0,
(1-\bar\alpha_t)I\right)
}.
```

It can be sampled directly without simulating the earlier timesteps:

```math
\boxed{
x_t=\sqrt{\bar\alpha_t}\,x_0
+\sqrt{1-\bar\alpha_t}\,\epsilon,
\qquad \epsilon\sim\mathcal N(0,I)
}.
```

## Inductive check

Assume

```math
x_{t-1}=\sqrt{\bar\alpha_{t-1}}x_0
+\sqrt{1-\bar\alpha_{t-1}}\epsilon.
```

Substitution into the one-step transition gives signal coefficient
`sqrt(alpha_t * bar(alpha)_{t-1}) = sqrt(bar(alpha)_t)`. The covariance of the
two independent noise contributions is

```math
\alpha_t(1-\bar\alpha_{t-1})I+(1-\alpha_t)I
=(1-\bar\alpha_t)I,
```

which proves the result by induction, with the `t=1` transition as the base
case.

## Interpretation

`bar(alpha)_t` is the retained signal power. Early in the chain it is close to
one, so `x_t` resembles the data. As it approaches zero, the conditional mean
vanishes and the covariance approaches `I`, making `x_t` approximately standard
Gaussian. The direct formula is the one used during DDPM training: sample a
timestep, sample one Gaussian noise tensor, construct `x_t`, and ask the model
to predict the injected noise.

## Code indexing convention

The derivation uses transitions numbered `1, ..., T`, with `x_0` denoting clean
data. The implementation stores the first transition at array index `0` and the
last at index `T-1`. Thus code timestep `0` applies `beta[0]`; it is very close
to clean for a small first beta, but it is not exactly the clean input. No fake
identity timestep is added.

For a batch of timesteps, `extract` gathers one coefficient per example and
reshapes `[B]` to `[B,1,1,1]`. Broadcasting then applies the selected scalar to
every channel and pixel of that example. `q_sample` implements the boxed direct
sampling equation in `src/diffusion_models/diffusion/ddpm.py`.

## Recovering the clean image when noise is known

The direct formula is algebraically invertible if the exact sampled noise is
available and `bar(alpha)_t > 0`:

```math
\boxed{
x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\,\epsilon}
{\sqrt{\bar\alpha_t}}
}.
```

This is implemented by `predict_x_start_from_noise`. It is not a practical
denoising method—the noise is unknown during generation—but it is a strong
implementation check. The milestone tests separately verify the schedule by
hand, the forward formula by hand, and the full algebraic round trip so that a
shared error in the forward and inverse functions cannot pass unnoticed.
