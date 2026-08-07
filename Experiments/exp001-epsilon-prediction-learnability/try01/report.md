# Try 01: Synthetic and Fixed-16 Learnability

## Goal and hypothesis

Test whether the existing smoke U-Net can optimize the explicit DDPM
epsilon-prediction objective under the two frozen gates in the authoritative
plan. The preregistered hypothesis is in `../hypothesis.md`.

## Configuration

See `docs/plans/training_and_overfit.md`. The baseline commit is
`26e735e597001251d92075d5bf02fb5649851800`; both runs are restricted to local
CPU execution.

## Changes

- Added explicit epsilon-batch construction, scalar MSE, checked optimizer
  update, global gradient norm/clipping, and immutable JSONL logging.
- Added synthetic-only unit tests and two frozen CPU experiment scripts.
- Added a durable fixed-16 manifest, raw logs, CSV loss data, summary JSON, and
  a 1,200×700 loss curve.
- Preserved the U-Net architecture and DDPM forward equations unchanged.

## Verification and execution

The full suite passed with 31 tests; Ruff lint, Ruff formatting, and
`git diff --check` passed. Gate A ran for 40 steps in 1.7245 seconds. Gate B ran
for 500 steps in 102.9107 seconds. Both used one local CPU thread and completed
normally after dataset staging/verification. Exact commands and hashes are in
the summary JSON files and `metadata.yaml`.

## Results, figures, and observations

| Gate | Initial loss | Early mean | Final loss | Late mean | Late/early | Result |
|---|---:|---:|---:|---:|---:|---|
| Fixed synthetic | 1.144742 | 0.773442 | 0.005317 | 0.008252 | 0.010670 | PASS |
| Fixed-16 CIFAR-10 | 1.094494 | 0.220184 | 0.052640 | 0.043873 | 0.199254 | PASS |

All 540 logged losses and gradient norms were finite. The fixed-16 run's global
gradient norm ranged from 0.049766 to 1.709329 with mean 0.122540. The curve was
visually inspected and agrees with the raw data: rapid early reduction followed
by lower stochastic fluctuations and occasional spikes.

The smoke model contains no attention. The separate attention diagnostic
confirmed first-pass zero QKV gradient and second-pass nonzero QKV gradient
after the zero-initialized output projection updated.

## Failure analysis and limitations

The experiment has no held-out evaluation, checkpoint, negative-control target,
or reverse sampler. The loss decrease therefore cannot establish generation
quality or generalization. The code ran from an uncommitted working tree, so
the summaries preserve source/config hashes in addition to baseline commit
`26e735e`.

The initial script invocation failed before execution because the editable
package path was not visible; the script bootstrap was fixed and all tests were
rerun. The first dataset attempt failed before loading because sandboxed DNS was
blocked. An approved official transfer was later interrupted at 2.6% because a
complete official cache already existed in the workspace. The staged archive
matched the official MD5 before the unchanged `--download` command verified it.

## Interpretation and next step

The evidence supports the preregistered conclusion that the complete CIFAR-10
epsilon-prediction pipeline can learn a fixed small dataset. It does not support
claims about sample quality, reverse-process correctness, generalization, or
full training. Stop at Milestone 3 pending review.

## Exact evidence paths

- Plan: `docs/plans/training_and_overfit.md`
- Metadata: `Experiments/exp001-epsilon-prediction-learnability/try01/metadata.yaml`
- Manifest: `configs/cifar10_overfit_16_manifest.json`
- Synthetic log: `Experiments/exp001-epsilon-prediction-learnability/try01/logs/synthetic_metrics.jsonl`
- Overfit log: `Experiments/exp001-epsilon-prediction-learnability/try01/logs/overfit_metrics.jsonl`
- Overfit CSV: `Experiments/exp001-epsilon-prediction-learnability/try01/results/overfit_loss.csv`
- Loss curve: `Experiments/exp001-epsilon-prediction-learnability/try01/figures/overfit_loss_curve.png`
- Milestone report: `Reports/milestone_03.md`
