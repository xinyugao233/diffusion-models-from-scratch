# State

- Status: Milestone 3 and `EXP001/try01` are `COMPLETED`; both frozen local CPU
  learnability gates passed.
- Baseline commit: `26e735e597001251d92075d5bf02fb5649851800`
  (`feat: add tested time-conditioned CIFAR-10 U-Net`).
- Remote state: private GitHub repository exists at
  `xinyugao233/diffusion-models-from-scratch`; Milestone 2 GitHub Actions passed
  on 2026-08-06 with 22 tests plus Ruff lint and format checks.
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
- Immediate next action: stop at Milestone 3 and await review or explicit
  authorization. No commit was created automatically.
