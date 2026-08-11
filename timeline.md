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

## 2026-08-07
- Created experiment scaffold `exp002-deterministic-checkpoint-resume` for `Deterministic checkpoint resume`.
- Added initial index entry for `EXP002`.

- 2026-08-07: Preserved `EXP002/try01` as failed after a comparator rejected
  equivalent `OrderedDict` and `dict` containers before comparing model tensors.
  Froze a mapping-aware comparison correction as `try02` without changing the
  scientific configuration or exact-equality rule.

- 2026-08-07: Completed `EXP002/try02`; uninterrupted 100-step and reconstructed
  50+50 CPU training matched exactly in model, EMA, optimizer, all losses,
  global step, and next RNG draw/loss. The no-RNG-restore control differed.

## 2026-08-07
- Created experiment scaffold `exp003-ddpm-reverse-sampling` for `DDPM reverse sampling`.
- Added initial index entry for `EXP003`.

- 2026-08-07: Published Milestone 4 as commit `2f13b3f`; push-triggered GitHub
  Actions run `31152749677` passed 41 tests plus Ruff lint and formatting.
- 2026-08-07: Completed `EXP003/try01`; independent posterior arithmetic and
  13 focused sampling tests passed, and the random EMA smoke U-Net completed a
  finite seeded 1,000-call DDPM chain with six visually inspected states. No
  training or image-quality evaluation was performed.

## 2026-08-07
- Created experiment scaffold `exp004-full-cifar-10-ddpm-training` for `Full CIFAR-10 DDPM training`.
- Added initial index entry for `EXP004`.

- 2026-08-07: Published Milestone 5 as commit `7061038`; GitHub Actions run
  `31154154963` passed 54 tests plus Ruff lint and formatting.
- 2026-08-07: Froze the `EXP004/try01` full CIFAR-10 training plan and
  implemented deterministic production orchestration plus Slurm-only execution
  scripts. Local validation reached 61 passing tests; no training was run.

## 2026-08-08

- Published the Hellbender portability repair as commit `7ad4524`; GitHub
  Actions run `31155794394` passed 61 tests plus Ruff lint and formatting.
- Preserved setup job `15914281`, which failed before environment creation
  because `SLURM_TMPDIR` was unset; repaired setup job `15914382` and Slurm
  checks job `15914409` completed.
- Gate A job `15914422` passed 20 real-data steps. Preserved Gate B attempt
  `15914462`, which was assigned an unsupported V100 before step 1.
- Gate B H100 jobs `15914482` and `15938195` passed the 250+250 new-process
  resume and exact repeated-sampling gate with 500 finite contiguous records.
- Completed full H100 job `15943245`: 50,000 finite contiguous records, ten
  checkpoints, fixed-seed 10k/25k/50k samples, loss ratio `0.465576`, and
  recognizable CIFAR-like structure at 50k. Closed `EXP004/try01` and
  Milestone 6 without launching a 100k extension or DDIM.

## 2026-08-12
- Created experiment scaffold `exp005-ddim-sampling-speed-quality-comparison` for `DDIM sampling speed-quality comparison`.
- Added initial index entry for `EXP005`.
- Implemented deterministic DDIM sampling, froze the four-setting comparison,
  and passed 71 local CPU tests plus Ruff and launcher checks. `EXP005/try01`
  is ready for exact-commit remote and Slurm validation.
