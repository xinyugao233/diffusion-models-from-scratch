# EMA and Reproducible Checkpoint/Resume Plan

## Status and authority

This is the authoritative implementation and execution plan for Milestone 4
and `EXP002/try01`. It was frozen on 2026-08-07 before EMA/checkpoint code was
implemented or the deterministic resume experiment was run.

Milestone 3 is closed at commit
`45c17b8a743b56b0233794dfbc25b8709863473c`. Push-triggered GitHub Actions run
`31151451025` passed on `main` in 49 seconds, and the worktree was clean before
the `EXP002` scaffold was created.

## Research question

Can the complete training state be saved at step `K`, reconstructed in fresh
Python objects, and resumed through step `N` without changing the deterministic
CPU optimization trajectory relative to uninterrupted training?

## Motivation and existing evidence

Milestone 3 establishes that the smoke U-Net and epsilon-prediction objective
are learnable. Longer future runs require trustworthy interruption recovery.
Model weights alone are insufficient because AdamW momentum, EMA weights,
global step, configuration, and random-number streams all influence later
updates.

## Hypothesis and interpretation

If the checkpoint stores and restores every trajectory-relevant state object,
then a 100-step uninterrupted path and a 50-step-save-reconstruct-restore-50-step
path will be bitwise identical on deterministic CPU execution. Omitting RNG
restoration should change the next sampled timestep/noise batch and therefore
fail the trajectory comparison.

Passing supports exact local CPU resumability for the frozen configuration. It
does not establish cross-device, cross-version, GPU, distributed, mixed-
precision, or full-dataset reproducibility.

## Baseline, intervention, and controls

- Baseline: Milestone 3 commit `45c17b8` with model, optimizer, and RNG state
  held only in memory.
- Intervention: explicit EMA plus structured training checkpoint save/load.
- Path A control: uninterrupted 100-step optimization.
- Path B intervention: save after step 50, destroy all model/optimizer/EMA
  objects, reconstruct, restore, and train to step 100.
- Negative control: reconstruct and load without restoring RNG; the next
  timestep/noise draw must differ from the correctly restored reference.

## Frozen scientific invariants

- The validated smoke U-Net and forward-diffusion equations remain unchanged.
- Images have shape `[4,3,32,32]`, dtype `float32`, and range `[-1,1]`.
- The linear schedule remains `T=1000`, beta `1e-4` to `2e-2`.
- The same epsilon-prediction training operation from Milestone 3 is used.
- EMA updates occur exactly once after each successful optimizer update.
- EMA parameters are never passed to the optimizer and never require gradients.
- Floating model parameters are exponentially averaged. Model buffers are
  copied exactly from the training model on every EMA update, not averaged.
- The checkpoint serializes state dictionaries, never whole Python model
  objects.
- The experiment uses CPU, one PyTorch thread, and deterministic algorithms.
- No dataset, GPU, cluster, sampling, DDIM, EMA warmup schedule, AMP,
  GradScaler, distributed training, or full CIFAR-10 training is permitted.

## Frozen experiment configuration

| Field | Value |
|---|---|
| Model | unchanged 491,107-parameter smoke U-Net |
| Model initialization seed | `41` |
| Fixed synthetic-image seed | `43` |
| Training torch RNG seed | `47` |
| Python RNG seed | `53` |
| Batch size | `4` |
| Device / dtype | CPU / `float32` |
| PyTorch threads | `1` |
| Optimizer | AdamW, betas `(0.9,0.999)`, eps `1e-8`, weight decay `0` |
| Learning rate | `1e-3` |
| Gradient clip | global L2 norm `1.0` |
| EMA decay | `0.99` |
| Total steps `N` | `100` |
| Interruption step `K` | `50` |
| Schedule | linear, `T=1000`, beta `1e-4` to `2e-2` |
| Baseline commit | `45c17b8a743b56b0233794dfbc25b8709863473c` |

Decay `0.99` is intentionally used for this short correctness experiment so
EMA smoothing is observable within 100 steps. The class supports other decay
values, including the larger values normally used for long diffusion runs.

Exact command:

```bash
.venv/bin/python scripts/run_resume_validation.py \
  --config configs/resume_validation.json
```

## EMA definition and buffer policy

For every named training parameter after optimizer step `s`:

```math
\theta_{ema,s}=\gamma\theta_{ema,s-1}+(1-\gamma)\theta_s.
```

EMA is initialized as an exact detached clone of model parameters and buffers.
Parameter names, shapes, devices, and dtypes must match on every update and
load. Buffers such as running statistics or counters are copied from the
training model on each update so future evaluation uses current non-optimized
state without applying an ill-defined exponential average to integer counters.
A `copy_to(model)` method will expose the complete EMA state for future
sampling without linking gradients.

## Checkpoint schema

Version `1` contains:

- `model_state_dict`
- structured `ema_state_dict`
- `optimizer_state_dict`
- nonnegative integer `global_step`
- JSON-compatible training `configuration`
- Python `random` state
- torch CPU RNG state
- optional named `torch.Generator` states when callers supply them
- CUDA RNG states only when explicit CUDA device indices are supplied
- NumPy RNG state only when NumPy randomness is explicitly enabled
- repository/version metadata

This experiment uses Python and torch CPU randomness only. NumPy is not used,
so NumPy state is deliberately absent rather than fabricated. CUDA is not used,
so no CUDA state is captured.

Checkpoint saving refuses to overwrite an existing file. Loading validates the
schema/version and restores model, EMA, optimizer, global step, configuration,
and RNG explicitly. Incomplete, corrupted, incompatible, or configuration-
mismatched checkpoints must fail clearly.

## Deterministic comparison protocol

1. Build fixed synthetic `x_start` without consuming the training RNG stream.
2. Path A: initialize identical components, seed training RNG, run 100 fresh-
   noise/timestep updates, and record final states and last loss.
3. Path B: reinitialize identically, seed training RNG, run 50 updates, save
   the checkpoint, destroy objects, reconstruct fresh objects, load and restore
   RNG, then run steps 51–100.
4. At step 50, temporarily sample the expected next batch and restore RNG to
   preserve Path B. Separately load without RNG restoration and verify its next
   batch differs.
5. After step 100, sample one next batch for each valid path and calculate its
   next loss without updating parameters.
6. Compare every model tensor, every EMA tensor, the complete nested optimizer
   state, global step, last loss, next timesteps, next noise tensor, and next
   loss.

## Acceptance criteria

All comparisons between Path A and the correctly resumed Path B must use exact
equality:

- `torch.equal` for every model, EMA, optimizer tensor, and next noise tensor;
- ordinary equality for tensor names, nested optimizer scalars/containers,
  global step, last loss, next timesteps, and next loss;
- EMA update count exactly `100`;
- checkpoint configuration exactly equal to the frozen JSON configuration;
- the negative control without RNG restoration must produce a different next
  timestep or noise tensor;
- all losses and gradients finite;
- normal exit and one complete summary record.

No tolerance fallback is preregistered. Any non-exact comparison fails this try
and must be diagnosed before acceptance.

## Unit and integration tests

Synthetic CPU-only tests will cover:

1. EMA initialization equals model parameters and buffers.
2. One scalar EMA update matches hand calculation.
3. Multiple EMA updates match hand calculation.
4. EMA tensors are detached and do not require gradients.
5. Optimized model and EMA parameters diverge.
6. EMA state save/load and `copy_to` are exact.
7. Checkpoint round trip restores model, EMA, optimizer, global step, config,
   and metadata.
8. Python and torch CPU RNG restoration.
9. Corrupted/incomplete schema rejection.
10. Incompatible model and configuration errors.
11. A short exact uninterrupted-versus-resumed trajectory.

Normal CI must remain dataset-free and fast; the 100-step smoke-U-Net
experiment is executed separately.

## Expected artifacts and counts

- `configs/resume_validation.json`: one frozen configuration.
- `EXP002/try01/checkpoints/resume_step_0050.pt`: one structured checkpoint.
- `EXP002/try01/results/resume_summary.json`: one comparison summary.
- No checkpoint or result file may be overwritten.
- No PDF is required unless later presentation work is explicitly requested;
  the structured result, Markdown reports, and tests are canonical for this
  correctness milestone.

## Failure and stop conditions

- Stop if Milestone 3 commit/CI/clean-baseline conditions are not satisfied.
- Stop on any nonfinite loss or gradient.
- Stop if RNG omission does not alter the negative-control next draw.
- Stop on any non-exact valid-path comparison.
- Do not weaken exact equality, change seeds/steps/decay, or add state after
  observing a mismatch within this try.
- Stop after Milestone 4; do not begin DDPM sampling.

## Documentation plan

Create `docs/understanding/ema_and_checkpoints.md` and
`Reports/milestone_04.md`; update the `EXP002` try record, project state,
experiment index, timeline, README, and project specification after validated
execution. Preserve source/config/checkpoint/result hashes and all setup
failures.

## Revision record

### 2026-08-07 — try02 comparator correction

`try01` completed but failed because the validation helper required identical
container classes before recursively comparing values. PyTorch returns an
`OrderedDict` from `model.state_dict()`, while the uninterrupted snapshot was a
plain dictionary of cloned tensors. Thus `model_state_exact=false` was emitted
without any model tensor comparison even though loss trajectory, EMA,
optimizer, RNG, next draw, and next loss were exact.

`try01` outputs are preserved immutable. `try02` changes only the comparison
implementation to treat any two `Mapping` subclasses equivalently when their
keys and recursively compared values are exact. All scientific settings and
the no-tolerance acceptance rule remain unchanged. Its exact command is:

```bash
.venv/bin/python scripts/run_resume_validation.py \
  --config configs/resume_validation_try02.json
```
