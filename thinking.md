# Thinking

- Begin with mathematical definitions and testable utilities before model code.
- Keep data in `[-1, 1]` for symmetric model inputs; invert exactly for figures.
- Milestone one uses the canonical linear schedule (`T=1000`, `1e-4` to
  `2e-2`) to validate the forward mechanics. Linear versus cosine remains an
  open controlled comparison for a future training experiment.
