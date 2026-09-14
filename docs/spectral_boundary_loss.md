# Optional spectral-boundary loss

## Implementation audit

The production entry point is `scripts/train_full_cifar10.py`, launched under
Slurm by `cluster/sbatch_train.sh`. It trains an unconditional CIFAR-10 DDPM
with a 1,000-step linear beta schedule. One timestep is sampled uniformly per
image in `sample_epsilon_training_batch`, and the forward input is

`x_t = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) epsilon`.

The U-Net predicts `epsilon`. The baseline objective in
`epsilon_prediction_loss` is mean squared error over all batch, channel, and
pixel coordinates. Images are RGB NCHW tensors scaled from `[0,1]` to `[-1,1]`.
Training is single-GPU float32 with AdamW, gradient clipping, and EMA; it does
not use mixed precision, DDP, or gradient accumulation. Timestep/noise draws,
data order, fixed evaluation noises, checkpoint names, append-only JSONL
metrics, and output manifests are deterministic and explicitly recorded.

The repository already uses complete orthonormal FFTs in
`diffusion_models.fourier`. The empirical spectrum reused here comes from the
validated E006 1K CIFAR-10 gate. It used an orthonormal `rfft2`, exact
half-plane Parseval weights, floor-radius bins, averaging over images, RGB
channels, and full-plane sites per shell. The copied CSV is accompanied by a
provenance record in `configs/`.

## Hook and objective

The optional hook is inside `production_train_step`, after the unchanged model
forward pass has produced the epsilon residual and before backpropagation. The
forward diffusion process, timestep sampling, target, model, optimizer, and
sampler are unchanged.

E006 uses additive coordinates `y = x_0 + sigma epsilon`, whereas the DDPM uses
variance-preserving coordinates. The compatible noise scale is therefore

`sigma_eff(t) = sqrt((1-alpha_bar_t) / alpha_bar_t)`.

For shell power `P_r`, the raw radial weight is

`a_{t,r} = w_floor + exp(-0.5 * (log(P_r)-2 log(sigma_eff(t)))^2 / tau^2)`.

The default normalization divides by the full-coefficient weighted mean,

`w_{t,r} = a_{t,r} / [sum_r n_r a_{t,r} / sum_r n_r]`,

where `n_r` is the number of full FFT sites in shell `r`. For epsilon residual
`e = epsilon_hat - epsilon`, the optimization loss is

`L_boundary = mean_{b,c,k} w_{b,r(k)} |FFT_ortho(e)_{b,c,k}|^2`.

This is deliberately coefficient-weighted rather than equally shell-averaged.
It changes the baseline only through the moving radial weights: when weights
are uniform, Parseval recovers the original pixel MSE. An equal-shell
reduction would additionally change every shell's total influence according to
its multiplicity and should be tested as a separate control.

When the feature is absent or `enabled` is false, `production_train_step`
uses the original loss tensor directly and performs no FFT. Existing configs,
metrics, checkpoint behavior, and gradient calculations remain unchanged.

## Configuration

Add this object to a copy of the baseline training JSON:

```json
"spectral_boundary_loss": {
  "enabled": true,
  "radial_power_path": "configs/cifar10_e006_radial_power.csv",
  "tau": 1.0,
  "weight_floor": 0.05,
  "normalization": "coefficient_mean",
  "fft_normalization": "ortho",
  "sigma_min": null,
  "sigma_max": null
}
```

Suggested initial configurations, to be frozen before any run:

- baseline: omit the object or set `enabled` to false;
- broad: `tau=1.0`, `weight_floor=0.05`, no sigma limits;
- narrow: `tau=0.35`, `weight_floor=0.05`, no sigma limits;
- wild-middle only: the broad or narrow setting with `sigma_min=0.14` and
  `sigma_max=8.4`.

These tau and floor values are starting points, not validated optima. The two
sigma limits must either both be numbers or both be null/omitted.

Enabled runs append both `weighted_loss` and `unweighted_loss`, the effective
sigma range, active-example fraction, mean peak shell, mean information radius,
mean/max spectral weight, and Kish effective shell count to the existing
per-step JSONL record. Existing metrics remain present.

## Scientific limits

The E006 spectrum is a dataset-level diagnostic from the first 1,000 CIFAR-10
training images, not a causal or model-side memorization result. E006 found
that the coefficient-level `R_MI(sigma)` contour did not trace the cumulative
CALM high-posterior/high-coverage region. This feature therefore tests a new
optimization hypothesis; it must not be described as targeting an established
memorization boundary or as reducing FLOPs.

The first training comparison should keep initialization, data order,
timestep/noise draws, optimizer, step count, evaluation seeds, and compute
matched. Report quality versus steps, examples, and wall time, plus a distinct
checkpoint-aligned memorization metric. A uniform-radial-weight control is the
smallest way to distinguish the moving frontier from generic spectral
reweighting; equal-shell, inverted, random-radial, fixed-low, and fixed-high
controls remain later work.

## Sanity tests

`tests/test_spectral_boundary.py` checks E006 shell multiplicities, DDPM-to-
additive sigma conversion, outward peak motion as sigma decreases, wild-middle
gating, Parseval recovery under uniform weights, narrow/broad and zero/positive
floor stability, exact disabled-path identity, and finite nontrivial weighted
loss. The existing training, checkpoint, Fourier, and full-pipeline unit tests
remain applicable.
