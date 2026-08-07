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

Milestone 4 is now complete locally as `EXP002/try02`. Explicit EMA and a
versioned structured checkpoint preserve model, EMA, AdamW, global step,
configuration, and Python/torch RNG. The corrected deterministic validation
found exact equality between 100 uninterrupted steps and reconstructed 50+50
training for every state tensor, the full loss trajectory, and the next random
draw/loss. The RNG-omission control differed as required. Try01's mapping-
container comparator failure is preserved rather than overwritten.

The valid conclusion is limited to exact resume on this deterministic local CPU
software stack. No reverse sampler or generation evaluation exists. Milestone
4 is uncommitted and the project stops here pending review or an explicit DDPM
sampling request.
