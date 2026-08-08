# State

- Status: Milestone 6 and `EXP004/try01` are `COMPLETED`; the frozen 50k
  full-CIFAR-10 DDPM run and all preflight gates passed.
- Baseline commit: `706103861c7d12ff3cb7dee037b6a10514b46b5e`
  (`feat: add tested ancestral DDPM sampling`).
- Remote state: private GitHub repository `xinyugao233/diffusion-models-from-scratch`
  has green push-triggered CPU checks for Milestone 5. Run `31154154963` passed
  on 2026-08-07 with 54 tests plus Ruff lint and formatting.
- Milestone 2 model state: the primary U-Net has 12,852,547 trainable
  parameters and the smoke U-Net has 491,107. The validated architecture and
  forward-diffusion equations are frozen for Milestone 3.
- Gate A: 40 finite records, loss `1.144742 -> 0.005317`, late/early mean ratio
  `0.010670`, parameters changed, duration `1.7245 s`, PASS. The separate
  attention diagnostic observed zero first-pass QKV gradient and nonzero
  second-pass QKV gradient.
- Gate B: fixed indices `0..15`, 500 finite records, loss
  `1.094494 -> 0.052640`, late/early mean ratio `0.199254`, duration
  `102.9107 s`, no instability, PASS. Device was one local Apple arm64 CPU
  thread; the official CIFAR-10 archive MD5 was
  `c58f30108f718f92721af3b95e74349a`.
- Validation: 31 tests pass; Ruff lint and formatting plus `git diff --check`
  pass; raw record counts and recomputed statistics were independently checked;
  the 1,200×700 RGB loss curve was visually inspected.
- Scientific conclusion: the complete epsilon-prediction pipeline can learn
  the fixed 16-image dataset. Sampling quality, generalization, reverse-process
  correctness, and full-dataset training remain untested.
- Maintenance debt: GitHub Actions emitted a non-blocking Node.js runtime
  deprecation warning. Updating action versions is deliberately deferred.
- Milestone 4 implementation: explicit detached EMA, exact-copy buffer policy,
  versioned structured checkpoints, strict restoration/errors, global-step,
  Python/torch/named-generator RNG support, and future EMA evaluation state.
- `EXP002/try01`: preserved `FAILED` because its comparator rejected equivalent
  `OrderedDict` and `dict` containers before tensor comparison; all other
  comparisons were exact.
- `EXP002/try02`: `COMPLETED`; all 12 exact comparisons passed for uninterrupted
  100-step versus reconstructed 50+50 training. The no-RNG-restore control
  changed the next draw. Checkpoint SHA-256 is
  `3d4ff5184457c4002b2f7a8bb192360e9391989bbe9f523c3c8cc30f474012c0`.
- Validation: 41 tests pass; Ruff lint/format and `git diff --check` pass;
  checkpoint schema/hash/config and scientific try invariants were independently
  audited. U-Net and forward-diffusion files remain unchanged.
- Scientific conclusion: exact resume is established on the frozen local CPU
  software stack only. Cross-device/version, GPU, distributed, full-data, and
  sampling behavior remain untested.
- Milestone 5 implementation: exact DDPM posterior, epsilon-to-clean
  reconstruction, fixed posterior variance, deterministic final step, supplied
  reverse noise, seeded complete loop, per-step finite checks, and selected
  trajectory capture.
- `EXP003/try01`: `COMPLETED`; 13 focused sampling tests and one full random EMA
  smoke chain passed. The CPU run made exactly 1,000 model calls in 12.6002
  seconds and produced a finite `[4,3,32,32]` final tensor plus six requested
  trajectory states.
- Validation: 54 tests pass; Ruff lint/format and `git diff --check` pass; the
  artifact hashes, schema, call count, shape, finiteness, and final-state
  identity were independently audited; the 632×454 RGB trajectory was visually
  inspected. U-Net, forward diffusion, and schedules remain unchanged.
- Scientific conclusion: DDPM ancestral sampling is structurally correct for
  the tested equations and CPU execution. Random-model visuals make no image-
  quality, trained-checkpoint, FID, likelihood, or generalization claim.
- Milestone 6 implementation: deterministic step-indexed full-data batches,
  explicit checked AdamW-then-EMA updates, append-only metrics, checkpoint and
  fixed-sample cadence, Slurm runtime refusal, CUDA provenance/memory metrics,
  two-process Gate B resume, and `$SLURM_TMPDIR`-safe cluster scripts.
- Validation: 61 tests pass; Ruff lint/format, shell syntax, and
  `git diff --check` pass. Tests are CPU-only and dataset-free.
- Run identity: commit `7ad4524fbbeb61fc78a024ee74e2638644012774`,
  GitHub Actions `31155794394`, config SHA-256 `8d4480ef...40487`, CIFAR-10
  archive MD5 `c58f30108f718f92721af3b95e74349a`.
- Execution: Gate A job `15914422`, Gate B jobs `15914482` and `15938195`, and
  full job `15943245` completed on H100. The full run produced exactly 50,000
  unique contiguous finite records, ten 5k-cadence checkpoints, and fixed-seed
  samples/loss curves at 10k, 25k, and 50k.
- Full result: first/last 1,000-step mean losses `0.06556945` and `0.03052753`,
  ratio `0.465576` versus frozen maximum `0.9`; final loss `0.03368700`;
  16.083 steps/s; 3.035 GB peak allocated memory. The inspected 50k EMA grid
  contains recognizable CIFAR-like animal and vehicle structure.
- Preserved infrastructure failures: setup job `15914281` exposed absent
  `SLURM_TMPDIR`; Gate B attempt `15914462` landed on an unsupported V100
  before step 1. Both are documented and neither changed scientific settings.
- Scientific conclusion: the complete DDPM trains stably on CIFAR-10, resumes
  across a new process, and produces recognizable EMA samples under ancestral
  sampling. FID/KID, held-out generalization, DDIM, and comparative claims
  remain untested.
- Immediate next action: close Milestone 6. No 100k extension or DDIM work is
  authorized without a new reviewed plan.
