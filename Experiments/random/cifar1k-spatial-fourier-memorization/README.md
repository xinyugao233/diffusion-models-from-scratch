# Random side experiment: CIFAR-1K spatial versus Fourier memorization

This is an independent exploratory experiment. It is deliberately outside the
repository's numbered milestone and `EXP###` sequence and makes no claim about
the Spectral CALM program.

## Question

Does the existing convolutional CIFAR-10 diffusion U-Net reach the same frozen
sample-level memorization threshold at a different number of optimizer steps
when the same first 1,000 CIFAR-10 training images are represented in pixel
coordinates versus lossless complex Fourier coordinates?

## Conditions

- Spatial: normalized RGB `x_0` with shape `[3,32,32]`.
- Fourier: `FFT2(x_0, norm="ortho")` exposed as interleaved
  `[Re R, Im R, Re G, Im G, Re B, Im B]` channels.

The Fourier condition uses the FFT of the paired RGB Gaussian noise. It never
draws six independent channels. Predictions and reverse states are projected
onto the Hermitian subspace, and clipping is performed in pixel space.

## Frozen primary protocol

- First 1,000 canonical CIFAR-10 training examples; no augmentation.
- Existing primary U-Net, base width 64; only input/output channels differ.
- Three paired seeds; same batches, timesteps, and RGB noise draws.
- AdamW, learning rate `2e-4`, EMA `0.9999`, batch size 128.
- Evaluate every 5,000 steps through a 100,000-step maximum.
- Generate 1,024 fixed-seed samples with deterministic DDIM-50.
- A sample is memorized when `d_1NN < d_2NN / 3` in normalized pixel L2.
- Threshold: at least 90% memorized samples at two consecutive checkpoints,
  with a fresh-panel confirmation at the first crossing.
- Stop a paired run only after both conditions cross or the maximum is reached.

Steps-to-threshold is primary. CUDA training time per condition and time to
threshold are reported separately. Unique training-neighbor coverage is not
confused with the sample-level memorization rate.

## Tries

- `try01`: authorized on 2026-08-31; configuration, Slurm jobs, raw metrics,
  visual audits, and reports are immutable once completed.
