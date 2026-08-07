# Thinking

- Begin with mathematical definitions and testable utilities before model code.
- Keep data in `[-1, 1]` for symmetric model inputs; invert exactly for figures.
- Milestone one uses the canonical linear schedule (`T=1000`, `1e-4` to
  `2e-2`) to validate the forward mechanics. Linear versus cosine remains an
  open controlled comparison for a future training experiment.
- Milestone 3 freezes a 20% early-to-late windowed loss reduction as the
  learnability threshold for both local CPU gates. This avoids requiring
  monotonic stochastic loss while preventing post-hoc threshold selection.
- GitHub Actions currently emits a non-blocking Node.js runtime deprecation
  warning. Updating action versions is maintenance debt and is intentionally
  deferred until after the training milestone.
- `EXP001/try01` passed both frozen learnability gates. The fixed-16 windowed
  loss reduction is strong evidence for correct end-to-end optimization wiring,
  but it cannot distinguish memorization from generalizable denoising and says
  nothing about reverse sampling quality.
- `EXP002/try02` establishes exact same-stack CPU resume for the frozen smoke
  trajectory. The failed try01 is a useful bookkeeping lesson: exact tensor
  comparison must not reject semantically equivalent mapping subclasses before
  examining keys and values.
- RNG restoration is empirically necessary: reconstruction without restoring
  the saved torch stream changed the immediate next timestep/noise draw.
- `EXP003/try01` separates sampler correctness from model quality. Independent
  posterior arithmetic, timestep-zero noise suppression, seed controls, and a
  complete 1,000-call run support the reverse implementation, while the
  noise-like random-model trajectory cannot evaluate learned generation.
- EMA stays outside sampler logic: a copied EMA model uses the ordinary model
  interface, preventing raw/EMA sampling code paths from diverging.
