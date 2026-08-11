# EXP005 Try 01: DDIM Sampling Speed–Quality Comparison

## Status

`SUBMITTED` — implementation, GitHub CI, and cluster checks pass. H100 job
`16409278` is pending for capacity; no checkpoint result exists yet.

## Goal

Compare DDPM-1000 with deterministic DDIM-100/50/25 using the same 50k EMA
checkpoint and fixed 16 initial tensors.

## Authoritative plan

`Experiments/exp005-ddim-sampling-speed-quality-comparison/plan.md`

## Configuration

Frozen settings are recorded in the plan and serialized in
`configs/ddim_comparison.json` (SHA-256
`80c6e27776ff6aa839145fa09cae006a17dbbc1a3ae3d138bf6379fe387a1108`).

## Verification

- `71 passed` with `.venv/bin/pytest -q`.
- Ruff lint and formatting checks passed.
- `bash -n cluster/sbatch_ddim_comparison.sh` passed.
- `python -m py_compile scripts/run_ddim_comparison.py` passed.
- `git diff --check` passed.
- GitHub Actions run `31545879080` passed on exact commit `20c6ea9`.
- Hellbender checks job `16409245` completed with 71 tests and both Ruff gates.

## Execution

H100 job `16409278` was submitted at cluster time `2026-08-11T18:20:43` from
full commit `20c6ea9fd8b558016a1e995abb7db0ea723962fc`. It requests one H100 for
30 minutes and stages to
`/home/xggh8/data/diffusion-models-from-scratch/exp005-ddim-comparison/try01`.
Its current state is `PENDING (Priority)`. Training and checkpoint extension
are not authorized.

## Results

No result exists.

## Figures

Pending fixed-seed four-setting comparison.

## Interpretation

No speed or quality claim is currently supported.

## Limitations

No FID/KID or distribution-level evaluation is planned.

## Exact evidence paths

- Parent checkpoint:
  `/home/xggh8/data/diffusion-models-from-scratch/exp004-full-cifar10-ddpm-training/try01/full/checkpoints/checkpoint_step_050000.pt`

## Next step

Monitor job `16409278` to completion, validate every expected artifact and
hash, and visually inspect all four fixed-seed grids.
