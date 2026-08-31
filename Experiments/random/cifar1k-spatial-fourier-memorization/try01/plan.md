# Try01 execution plan

## Hypothesis

Changing only the lossless representation presented to the existing
convolutional U-Net changes paired optimizer steps to a frozen memorization
threshold.

## Falsification criterion

The hypothesis is not supported if the paired seed differences in
steps-to-threshold are inconsistent or indistinguishable, or if any FFT/IFFT,
Parseval, Hermitian, paired-noise, or sampling gate fails. A time-only
difference with similar step counts is a compute-cost result.

## Execution

1. Generate the local visual audit for the exact first 1,000 images.
2. Publish the clean experiment commit.
3. Pull the commit into the Hellbender execution checkout.
4. Submit the Slurm CPU check job and repair until it passes.
5. Submit the three-task H100 paired-seed array.
6. Monitor every attempt to completion; preserve failed attempts.
7. Aggregate steps, CUDA time, memorization curves, unique-neighbor coverage,
   sample grids, implementation gates, and right-censoring.
8. Store a report in this try without adding the side experiment to the main
   numbered experiment index.
