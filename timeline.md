# Timeline

- 2026-08-04: Created the repository scaffold and initial data/theory milestone.
- 2026-08-04: Passed four data utility tests and Ruff; generated and inspected
  the deterministic 8×8 CIFAR-10 training grid.
- 2026-08-04: Implemented and validated the linear DDPM schedule, closed-form
  forward sampler, and exact-noise reconstruction; generated and inspected the
  fixed-noise forward-process grid.
- 2026-08-06: Implemented and CPU-validated the frozen time-conditioned
  CIFAR-10 U-Net, added CPU CI and empirical forward-moment validation, and
  recorded primary/smoke parameter counts. No training was started.

## 2026-08-06
- Created experiment scaffold `exp001-epsilon-prediction-learnability` for `Epsilon prediction learnability`.
- Added initial index entry for `EXP001`.

- 2026-08-06: Completed `EXP001/try01`; 40-step synthetic and 500-step fixed-16
  CIFAR-10 CPU gates passed their frozen loss-reduction criteria with finite
  logs. Recorded the exact manifest, hashes, raw losses, and inspected curve.
