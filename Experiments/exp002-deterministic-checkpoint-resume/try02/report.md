# Try 02: Mapping-Aware Exact Resume Comparison

## Goal and relation to try01

Repeat the exact same scientific experiment after correcting only the state
comparison implementation. `try01` compared container types before values and
therefore declared `OrderedDict(model.state_dict())` unequal to an equivalent
plain cloned dictionary without comparing any model tensor.

## Frozen configuration and invariants

Seeds, model, data, optimizer, schedule, EMA decay, steps, interruption point,
device, dtype, gradient clipping, exact-equality acceptance, and RNG-omission
negative control are unchanged. Only output paths move to `try02` and the
comparator accepts different `Mapping` subclasses while still requiring exact
keys, nested types, scalars, and tensor values.

## Verification and execution

The mapping-aware regression test and full repository suite passed with 41
tests; Ruff lint/format and `git diff --check` passed. The frozen command ran
Path A for 100 steps and Path B for 50+50 steps on one CPU thread. Path A took
4.5929 seconds and Path B, including checkpoint/reconstruction validation, took
4.6884 seconds.

## Results and interpretation

`COMPLETED`. All 12 exact comparisons passed: model, EMA, AdamW, global step,
EMA update count, full loss trajectory, final loss, next Python draw, next
timesteps, next noise, next loss, and the required RNG-omission divergence.

Both paths ended with loss `0.16175299882888794`. Their next loss was exactly
`0.41868576407432556`, next timesteps were `[12,899,157,802]`, and next noise
SHA-256 was
`79d7a08f86fbd37b56a9b713e4f2972f2567eeb6fa1c42df8d967dba765b3260`.

The step-50 checkpoint contains the complete versioned schema and has SHA-256
`3d4ff5184457c4002b2f7a8bb192360e9391989bbe9f523c3c8cc30f474012c0`.
The evidence supports exact same-stack deterministic CPU resume only.

## Evidence paths

- Plan revision: `docs/plans/ema_and_checkpoints.md`
- Config: `configs/resume_validation_try02.json`
- Metadata: `Experiments/exp002-deterministic-checkpoint-resume/try02/metadata.yaml`
- Checkpoint: `Experiments/exp002-deterministic-checkpoint-resume/try02/checkpoints/resume_step_0050.pt`
- Summary: `Experiments/exp002-deterministic-checkpoint-resume/try02/results/resume_summary.json`
- Milestone report: `Reports/milestone_04.md`
