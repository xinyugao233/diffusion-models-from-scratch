# Try 01: Exact Deterministic Resume

## Goal and hypothesis

Compare uninterrupted 100-step CPU training with a 50-step checkpoint,
complete object reconstruction, restoration, and continuation through step
100. The preregistered hypothesis requires bitwise equality throughout.

## Configuration

See the authoritative `docs/plans/ema_and_checkpoints.md`. Baseline commit is
`45c17b8a743b56b0233794dfbc25b8709863473c` with green remote CI run
`31151451025`.

## Changes

Implemented explicit EMA, structured state-dict checkpoints, Python/torch RNG
capture and restoration, strict checkpoint validation, and the frozen
uninterrupted-versus-resumed script.

## Verification and execution

The 40-test suite, Ruff lint/format, and `git diff --check` passed before the
run. The frozen command completed both 100-step paths and wrote one 7,983,729-
byte step-50 checkpoint plus one summary.

## Results and observations

`FAILED` under the exact acceptance rule. EMA, optimizer, final loss, entire
loss trajectory, global step, EMA update count, next Python draw, next
timesteps, next noise, and next loss were exact. The RNG-omission negative
control correctly changed the next draw. Only `model_state_exact` was false.

## Failure analysis and limitations

The failure was in validation bookkeeping: `clone_model_state` produced a plain
`dict`, PyTorch returned an `OrderedDict`, and `nested_exact` rejected differing
container classes before visiting any tensor. This does not justify passing
try01. The output is preserved, and try02 corrects only Mapping-subclass
handling while retaining exact recursive tensor equality.

## Interpretation and next step

Run try02 with identical scientific settings. No resumability conclusion is
accepted until the corrected exact comparator passes.

## Evidence paths

- Plan: `docs/plans/ema_and_checkpoints.md`
- Metadata: `Experiments/exp002-deterministic-checkpoint-resume/try01/metadata.yaml`
- Checkpoint: `Experiments/exp002-deterministic-checkpoint-resume/try01/checkpoints/resume_step_0050.pt`
- Summary: `Experiments/exp002-deterministic-checkpoint-resume/try01/results/resume_summary.json`
