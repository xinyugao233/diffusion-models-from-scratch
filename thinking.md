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
