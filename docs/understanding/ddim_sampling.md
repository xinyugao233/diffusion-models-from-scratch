# DDIM sampling: interview notes

## Why can a DDPM-trained model use DDIM at inference time?

Both samplers use the same learned epsilon predictor and the same forward
marginals. DDIM defines a different reverse trajectory whose training objective
is unchanged, so no retraining is needed.

## What does eta do?

`eta` controls the stochastic standard deviation in the DDIM transition.
`eta=0` removes the added reverse noise, making the trajectory deterministic
given its initial Gaussian tensor. This repository implements only that
deliberately narrow case.

## Why predict x0 during every step?

The model predicts epsilon, but the transition is most readable after solving
the forward equation for an estimate of the clean image. The update then mixes
that estimate with the predicted noise at the next selected noise level.

## Why is the final previous timestep represented by -1?

It is a code sentinel for the clean endpoint, where `alpha_bar=1`. Substituting
that value makes the noise coefficient zero and returns the clean estimate.
It is not an index into the stored schedule.

## What is the key implementation invariant?

The number of selected timesteps must equal the number of denoiser calls.
Endpoints must be preserved, the order must be strictly descending, and no
extra model evaluation may be hidden in bookkeeping.

## Why use the same initial noise for every setting?

It controls the starting latent so visible changes are attributable to the
reverse trajectory and step budget rather than a different random draw. DDPM
still needs a separately fixed reverse-noise seed because its ancestral steps
are stochastic.

## What does a 16-image grid fail to measure?

It does not measure population-level fidelity or diversity and cannot support
FID/KID claims. It is useful for transparent paired inspection and portfolio
evidence, not for a broad scientific ranking of samplers.
