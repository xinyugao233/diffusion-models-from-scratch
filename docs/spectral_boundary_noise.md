# Spectral-boundary weighted-noise diffusion

This study is separate from the frozen spectral-boundary weighted-loss study.
It changes the forward Gaussian chain and uses ordinary spatial epsilon MSE.

## Repository baseline

For code timestep `t` in `0,...,T-1`, the repository treats `beta[t]` as the
first through T-th forward transition. Thus code `t=0` denotes mathematical
`x_1`, while the clean endpoint `x_0` has cumulative attenuation one and is not
stored in the schedule. The baseline uses

\[
\alpha_t=1-\beta_t,\qquad
\bar\alpha_t=\prod_{s=0}^{t}\alpha_s,
\]

and

\[
q(x_t\mid x_0)=\mathcal N(\sqrt{\bar\alpha_t}x_0,
 (1-\bar\alpha_t)I).
\]

Training samples one code timestep uniformly, draws spatial white
\(\epsilon\sim\mathcal N(0,I)\), forms
\(x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon\), and minimizes
ordinary spatial \(\lVert\epsilon-\epsilon_\theta(x_t,t)\rVert_2^2\).
The current ancestral sampler reconstructs a clipped \(x_{0,\theta}\), inserts
it into the exact scalar DDPM posterior below, and adds posterior Gaussian
noise at every code timestep except zero. The current DDIM path uses the same
epsilon reconstruction with an endpoint-preserving descending subset of code
timesteps. Neither scalar path is used by the weighted-noise treatment.

## Hazard audit and correction

The proposed construction starts with \(h_t^{base}=-\log\alpha_t\) and

\[
b_{t,r}=w_{floor}+\exp\left[-\frac12
\left(\frac{\log P_r-\log\sigma_{eff}(t)^2}{\tau}\right)^2\right].
\]

The literal proposal `raw_h = h_base * b` is positive, monotone in cumulative
corruption, terminal-matched after normalization, and therefore defines a valid
Gaussian Markov chain. It does not, however, achieve the intended intervention
for the selected CIFAR-10 spectrum: because the baseline hazard rises strongly
and the floor is nonzero, shells 8 through 22 have their largest absolute
hazard at the final timestep rather than near their crossing.

The smallest correction used here allocates the fully redistributed hazard in
proportion to the boundary density itself:

\[
h^{boundary}_{t,r}=H_{base}\frac{b_{t,r}}{\sum_s b_{s,r}},\qquad
H_{base}=\sum_t h_t^{base}.
\]

A fixed interpolation remains available:

\[
h_{t,r}=(1-\rho)h_t^{base}+\rho h^{boundary}_{t,r}.
\]

Then \(\alpha_{t,r}=e^{-h_{t,r}}\),
\(\beta_{t,r}=1-\alpha_{t,r}\), and
\(\bar\alpha_{t,r}=\prod_{s\le t}\alpha_{s,r}\). Every hazard is nonnegative,
so cumulative corruption is monotone. Both components sum to the same
\(H_{base}\), so every shell has the baseline terminal attenuation. `rho=0`
is explicitly short-circuited to the stored baseline schedule.

## Forward marginal and epsilon objective

For Fourier coefficient `k` in radial shell `r(k)`:

\[
\hat x_t(k)=\sqrt{\bar\alpha_{t,r(k)}}\hat x_0(k)
+\sqrt{1-\bar\alpha_{t,r(k)}}\hat\epsilon(k).
\]

The implementation starts with real spatial white noise and uses
`rfft2/irfft2` with orthonormal normalization. Real radial multipliers preserve
the Hermitian constraints, and `irfft2` structurally returns a real tensor.

Ordinary spatial epsilon MSE is a consistent regression objective: the target
is the same underlying spatial white-noise draw and the optimum is
\(E[\epsilon\mid x_t,t]\). Parseval makes spatial and full Fourier epsilon MSE
equivalent. The score conversion is no longer scalar: if `B_t` is the Fourier
diagonal square root of \(1-\bar\alpha_{t,r}\), then
\(\nabla_{x_t}\log q_t(x_t)=-B_t^{-T}E[\epsilon\mid x_t]\). Consequently the
ordinary scalar reverse schedule cannot be used.

## Matched reverse chain

Each real Fourier degree of freedom follows

\[
q(\hat x_t\mid\hat x_{t-1})=
\mathcal N(\sqrt{\alpha_{t,r}}\hat x_{t-1},\beta_{t,r}).
\]

Writing \(\bar\alpha_{t-1,r}=1\) at code timestep zero, its exact posterior is

\[
\tilde\beta_{t,r}=\beta_{t,r}
\frac{1-\bar\alpha_{t-1,r}}{1-\bar\alpha_{t,r}},
\]

\[
\tilde\mu_{t,r}=
\frac{\sqrt{\bar\alpha_{t-1,r}}\beta_{t,r}}
{1-\bar\alpha_{t,r}}\hat x_0+
\frac{\sqrt{\alpha_{t,r}}(1-\bar\alpha_{t-1,r})}
{1-\bar\alpha_{t,r}}\hat x_t.
\]

The model predicts spatial epsilon. It is transformed to Fourier space and

\[
\hat x_{0,\theta}=
\frac{\hat x_t-\sqrt{1-\bar\alpha_{t,r}}\hat\epsilon_\theta}
{\sqrt{\bar\alpha_{t,r}}}.
\]

The posterior mean uses this estimate and reverse noise is generated as real
spatial white noise, transformed, scaled by
\(\sqrt{\tilde\beta_{t,r}}\), and inverse transformed. At code timestep zero
the posterior variance is zero. Sampling begins from spatial standard Gaussian
noise; the shared terminal attenuation matches the ordinary DDPM approximation.
