# Current Context

Milestones 1 and 2 are complete at baseline commit
`26e735e597001251d92075d5bf02fb5649851800`. The private GitHub repository and
remote CPU CI are operational; the Milestone 2 workflow passed 22 tests plus
Ruff lint and formatting. Its Node.js deprecation warning is recorded as
deferred maintenance debt.

Milestone 3 and `EXP001/try01` are complete. On one local CPU thread, the fixed
synthetic gate achieved late/early mean-loss ratio `0.010670` over 40 steps and
the frozen 16-image CIFAR-10 gate achieved `0.199254` over 500 steps; both beat
the preregistered maximum ratio `0.80`, produced only finite records, and
completed normally. The fixed subset, hashes, raw logs, loss CSV, visually
inspected curve, and exact environment are durable under the try folder and
summarized in `Reports/milestone_03.md`.

Milestone 4 is published at commit `2f13b3f` and GitHub Actions run
`31152749677` passed 41 tests plus lint and formatting. Explicit EMA and a
versioned structured checkpoint preserve model, EMA, AdamW, global step,
configuration, and Python/torch RNG. The corrected deterministic validation
found exact equality between 100 uninterrupted steps and reconstructed 50+50
training for every state tensor, the full loss trajectory, and the next random
draw/loss. The RNG-omission control differed as required. Try01's mapping-
container comparator failure is preserved rather than overwritten.

Milestone 5 and `EXP003/try01` are complete locally. The exact posterior,
epsilon-to-clean conversion, deterministic final step, supplied reverse noise,
seeded complete loop, and EMA-model interface pass 13 focused tests. One
random-weight CPU smoke run completed exactly 1,000 calls in 12.6002 seconds
with a finite four-image result and six visually inspected trajectory states.

The valid Milestone 5 conclusion is limited to structural DDPM sampling
correctness.
The noise-like figure is expected from random weights and provides no evidence
about generation quality. No training, trained-checkpoint evaluation, FID,
DDIM, GPU, or dataset work was performed. It was published as commit `7061038`,
and GitHub Actions run `31154154963` passed 54 tests plus lint and formatting.

Milestone 6 and `EXP004/try01` are now `READY` locally. The authoritative plan
freezes the existing primary U-Net, linear 1,000-step schedule, AdamW at
`2e-4`, batch 128, EMA `0.9999`, 20-step Gate A, two-job 250+250 Gate B, and an
initial 50k maximum. Seven new orchestration tests bring the local suite to 61
passes. No Slurm job or full-data optimization has run; the next action is to
publish the exact run commit and submit infrastructure checks before Gate A.
