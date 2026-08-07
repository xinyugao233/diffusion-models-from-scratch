# EXP004 Try 01: Full CIFAR-10 DDPM Training

## Goal

Validate the real-data production path and train the first complete CIFAR-10
DDPM baseline through at most 50,000 optimization steps.

## Hypothesis tested

The unchanged primary U-Net, linear DDPM, epsilon objective, EMA, checkpoint,
and sampler will train stably together and produce increasingly structured
fixed-seed EMA samples.

## Changes

Implemented production orchestration, deterministic step-indexed batches,
Slurm entrypoints, append-only logging, real-run checkpoint/resume, fixed-seed
EMA evaluation, provenance manifests, and artifact staging.

## Configuration

Authoritative plan: `docs/plans/full_cifar10_training.md`.
Frozen config: `configs/full_cifar10_training.json`.

## Verification

Local CPU validation: 61 tests passed, including seven new orchestration tests.
Ruff lint/format, shell syntax, and `git diff --check` pass. No dataset was
loaded and no training was run by the test suite.

## Execution

The exact clean implementation commit
`7a11b6e79d5828f68a0c7055069e95c56741052b` passed remote CI and was synced to
a detached Hellbender checkout. Environment setup job `15914281` failed in one
second, before environment creation or dependency installation, because this
cluster does not export `SLURM_TMPDIR`. No GPU was requested and training did
not begin.

## Results

No experiment result exists. Gate A expects 20 records, Gate B expects 500
records, and Gate C expects 50,000 records only after both preflights pass.

## Figures and tables

Pending. Gate B will create a structural four-image smoke grid; Gate C will
create fixed-seed 10k/25k/50k progression grids if authorized gates pass.

## Observations

None yet.

## Failure analysis

Job `15914281` exited `1:0` at line 10 of `cluster/sbatch_setup_env.sh` under
`set -u`: `SLURM_TMPDIR: unbound variable`. The same assumption existed in all
three Slurm entrypoints. The repair uses `$SLURM_TMPDIR` when available and a
job-specific `/tmp/$USER/ddpm-$SLURM_JOB_ID` directory otherwise. It does not
change scientific configuration or training code.

## Interpretation

No end-to-end claim is currently supported.

## Limitations

No FID/KID, DDIM, hyperparameter sweep, AMP, distributed training, or second
dataset is included.

## Next step

Validate and publish the Slurm portability repair, sync the new exact commit,
then resubmit environment setup and dataset-free checks before Gate A.

## Exact evidence paths

- Plan: `docs/plans/full_cifar10_training.md`
- Config: `configs/full_cifar10_training.json`
- Persistent remote output:
  `/home/xggh8/data/diffusion-models-from-scratch/exp004-full-cifar10-ddpm-training/try01/`
- Failed setup log:
  `/home/xggh8/data/diffusion-models-from-scratch/slurm-logs/setup-15914281.out`
