# State

- Status: Day 2 time-conditioned U-Net milestone implemented and locally
  validated; changes are uncommitted.
- Baseline commit: `6183fef` (`Milestone 1: implement DDPM forward process`).
- Implemented: primary and smoke U-Net configurations, sinusoidal and learned
  timestep embeddings, residual/downsample/upsample/attention blocks, complete
  encoder-bottleneck-decoder model, CPU CI, empirical forward moment check, and
  interview-oriented documentation.
- No training experiment or compute job exists.
- Training, sampling, checkpointing, EMA, and GPU/external compute are not
  authorized.
- Validation: 22 CPU tests pass; Ruff lint and format checks pass; primary model
  has 12,852,547 trainable parameters and smoke model has 491,107; both complete
  finite CPU forward/backward passes with `.grad` present and finite for every
  trainable parameter. On the first backward pass, zero-initialized attention
  output projections receive nonzero gradients while upstream attention
  gradients are present but zero-valued.
- CPU CI is configured but has not run remotely.
- Immediate next action: commit and push this milestone, then require green
  remote CI before beginning Milestone 3.
