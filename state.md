# State

- Status: Milestone 3 is remotely closed; Milestone 4 and `EXP002/try02` are
  `COMPLETED` locally under `docs/plans/ema_and_checkpoints.md`.
- Baseline commit: `45c17b8a743b56b0233794dfbc25b8709863473c`
  (`feat: add DDPM epsilon-prediction training and overfit validation`).
- Remote state: private GitHub repository `xinyugao233/diffusion-models-from-scratch`
  has green push-triggered CPU checks for Milestone 3. Run `31151451025` passed
  in 49 seconds on 2026-08-07 after commit `45c17b8` was pushed to `main`.
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
- Immediate next action: stop at Milestone 4 and await review or explicit DDPM
  sampling authorization. Milestone 4 remains uncommitted.
