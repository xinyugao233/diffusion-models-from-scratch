# Milestone 06 Report: First Full CIFAR-10 DDPM Training

## Current status

`READY` — authoritative plan, frozen configuration, production trainer, Slurm
entrypoints, and local orchestration tests are complete. No training or Slurm
job has been executed yet.

## Objective

Train the unchanged validated DDPM system on all 50,000 CIFAR-10 training
images and determine whether fixed-seed EMA samples become visibly structured
or recognizable through a maximum initial budget of 50,000 steps.

## Baseline and remote gate

- Milestone 5 commit:
  `706103861c7d12ff3cb7dee037b6a10514b46b5e`
- Milestone 5 GitHub Actions run: `31154154963`, SUCCESS, 1 minute 9 seconds
- Remote results: 54 tests, Ruff lint, and Ruff formatting passed
- Worktree was clean before `EXP004` creation

## Implemented orchestration

- Deterministic global-step-to-CIFAR-batch mapping with exact resume position.
- Explicit epsilon training step with checked gradients, clipping, AdamW, then
  EMA in the frozen order.
- Append-only JSONL metrics with contiguous-step resume validation.
- Frozen checkpoint and sampling cadence helpers.
- Fixed independently seeded initial sample tensors.
- Full Slurm-only trainer with source/config/dataset guards.
- Complete checkpoint restoration in a new process and CUDA RNG support.
- Periodic EMA sampling through the validated ancestral sampler.
- Per-segment summaries/manifests, hashes, throughput, runtime, and peak CUDA
  memory.
- Slurm setup, dataset-free checks, and GPU train scripts with `$SLURM_TMPDIR`
  cache isolation and persistent artifact staging.

No forward/reverse mathematics, U-Net architecture, EMA mathematics, or
checkpoint schema was redesigned.

## Local verification

```text
.venv/bin/python -m pytest -q
  -> 61 passed
.venv/bin/ruff check .
  -> All checks passed
.venv/bin/ruff format --check .
  -> 70 files already formatted
bash -n cluster/sbatch_setup_env.sh cluster/sbatch_checks.sh \
        cluster/sbatch_train.sh
  -> passed
git diff --check
  -> passed with no output
```

The seven new tests are CPU-only and dataset-free. They cover deterministic
resume batches, one real optimizer/EMA step, update order, cadence, fixed seeds,
append-only log recovery, structured checkpoint wiring, and Slurm refusal.

## Frozen identities before execution

- Configuration SHA-256:
  `8d4480ef52c2d299320cc374ae893b07a16690e091c88f303d1b9c8bf0140487`
- Orchestration source SHA-256:
  `54e6cc9f88c7602ab6f7622d16df6a1d7f235c4d5de4d03003286ccbb1a3f3fd`
- Training entrypoint SHA-256:
  `8aa1aa79db88d25aaf9317f59ccbad262c934a57f14c4e255b55e9fe11f3e81f`
- Slurm training script SHA-256:
  `53786c6df4ab76ec88fb6711d8ff7bf48c753c23c99d7b7dceb61156316e1033`

These source hashes will be recomputed after final formatting and before the
run commit is published. Job manifests, not this pre-execution draft, are the
authority for executed code identity.

## Planned gates

- Infrastructure: NOT RUN
- Gate A, 20 steps: NOT RUN
- Gate B, 250+250 resume and EMA sample: NOT RUN
- Gate C, maximum 50k: NOT RUN

No loss, throughput, memory, checkpoint, or sample result exists yet.

## Current valid conclusion

The production orchestration is locally testable and ready to publish for
Slurm preflight. The complete system has not yet trained on CIFAR-10, so no
end-to-end training or image-quality conclusion is supported.

## Evidence paths

- Plan: `docs/plans/full_cifar10_training.md`
- Understanding: `docs/understanding/full_training.md`
- Config: `configs/full_cifar10_training.json`
- Trainer: `scripts/train_full_cifar10.py`
- Orchestration: `src/diffusion_models/full_training.py`
- Slurm instructions: `cluster/README.md`
- Experiment record: `Experiments/exp004-full-cifar-10-ddpm-training/try01/report.md`
