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
- Milestone 6 deliberately freezes orchestration rather than redesigning
  validated components. Deterministic global-step batches make resume position
  explicit, while fixed initial tensors separate checkpoint progression from
  sampling-seed variation.
- A decreasing epsilon loss is necessary evidence of optimization but remains
  insufficient for generation. The 10k/25k/50k EMA progression is the first
  end-to-end qualitative gate, and FID/DDIM remain deferred.
- `EXP004/try01` confirms that distinction empirically: the frozen quantitative
  loss ratio passed (`0.465576`), while the fixed-seed grids provide separate
  qualitative evidence of progression from texture at 10k to recognizable
  CIFAR-like forms at 50k. Neither observation substitutes for FID or held-out
  evaluation.
- Cluster hardware identity is part of executable provenance. A memory-based
  GPU requirement was insufficient because the pinned CUDA 13 wheel supported
  H100 but not the scheduler-assigned V100; future Slurm plans should constrain
  a tested GPU type or pin a wheel whose architecture set covers every allowed
  device.
- `EXP005/try01` confirms the expected cost reduction on the frozen H100 setup:
  measured DDIM speedups closely track reduced NFE without being inferred from
  NFE ratios. The fixed samples remain recognizable even at 25 calls, so the
  preregistered monotonic visual-degradation expectation is not clearly visible
  in only 16 seeds. That absence must not be upgraded into a quality-equivalence
  claim without distribution-level metrics.
- Pairing is strongest when the run stores the materialized initial tensor
  identity, not only seed labels. EXP005 uses one tensor object across every
  condition, but its missing run-recorded `x_T` hash remains a bookkeeping
  limitation to fix in future comparison infrastructure.
