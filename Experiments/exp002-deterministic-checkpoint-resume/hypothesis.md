# Hypothesis

## Primary hypothesis

Restoring the complete structured checkpoint at step 50 will make every state
and subsequent stochastic draw exactly equal to uninterrupted 100-step CPU
training.

## Motivation and predicted result

The optimizer update is deterministic once model, optimizer, input, and random
draws are fixed. EMA is also deterministic. Therefore exact state and RNG
restoration should reproduce every later update bit-for-bit on the same CPU
software stack.

## Alternative explanations and controls

- Reconstructing objects consumes random numbers; restoring RNG after
  reconstruction controls this.
- Model weights may match while AdamW state differs; nested optimizer state is
  compared exactly.
- Final weights may accidentally match without correct future randomness; the
  next timestep/noise draw and next loss are compared.
- A test may never exercise RNG dependence; an omission negative control must
  demonstrably differ.

## Required controls

Identical initialization/data, fixed configuration, exact nested state
comparison, object destruction/reconstruction, post-load RNG restoration, and
the explicit no-RNG-restore negative control.

## Falsification and interpretation

Any non-exact valid-path state, loss, or next-draw comparison falsifies the
hypothesis for `try01`. Passing supports same-stack deterministic CPU resume
only, not portability across hardware or library versions.

## Stop conditions and preregistration

Do not introduce tolerances or change the frozen configuration after observing
results. Preregistered on 2026-08-07 in
`docs/plans/ema_and_checkpoints.md` before implementation.
