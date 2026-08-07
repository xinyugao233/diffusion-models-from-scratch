# Full CIFAR-10 DDPM Training Plan

## Status and authority

This is the authoritative implementation and execution plan for Milestone 6
and `EXP004/try01`. It was frozen on 2026-08-07 before production-trainer code
was implemented or any real-data optimization step was run.

Milestone 5 is closed at commit
`706103861c7d12ff3cb7dee037b6a10514b46b5e`. Push-triggered GitHub Actions run
`31154154963` passed with 54 tests plus Ruff lint and formatting, and the local
worktree was clean before `EXP004` was created.

## Research question

Can the complete validated DDPM system train stably on all 50,000 CIFAR-10
training images and produce visibly structured or recognizable samples from
EMA checkpoints under the validated 1,000-step ancestral sampler?

## Motivation and existing evidence

Milestones 1–5 separately validate forward diffusion, the primary and smoke
U-Nets, epsilon-prediction optimization, EMA/checkpoint resume, and ancestral
sampling. The remaining end-to-end uncertainty is orchestration on real data:
data ordering, long-running optimization, checkpoint recovery, EMA evaluation,
and fixed-seed checkpoint progression.

This experiment is a practical baseline for a public learning repository. It
is not a new diffusion formulation and does not test Spectral CALM hypotheses.
The separate shared research hub contains unrelated uncommitted `exp005` work
and will not be modified by this experiment.

## Hypothesis and allowed interpretation

If the validated components are connected without changing their mathematics,
then the primary U-Net will optimize stably on CIFAR-10, the late training loss
will be lower than the early loss, checkpoint/resume will preserve a valid
trajectory, and fixed-seed EMA samples will become less noise-like and more
object-structured across 10k, 25k, and 50k steps.

Passing supports only:

> The complete DDPM implementation trains stably on CIFAR-10 and its EMA
> checkpoints produce increasingly structured or recognizable samples under
> the validated ancestral sampler.

It does not establish state-of-the-art quality, competitive FID/KID,
superiority over another method, DDIM performance, or generalization
guarantees.

## Workflow choice

This is an implementation-driven validation experiment. Mathematical objects
remain frozen; the intervention is production orchestration around them. All
heavy work runs through Slurm on Hellbender. Local work is limited to source,
documentation, unit tests, and artifact inspection.

## Baseline, intervention, and controls

- Baseline: validated Milestone 5 commit `7061038` with no full-data trainer.
- Intervention: deterministic real-data batching, production optimization,
  structured logging, staged checkpointing/resume, and periodic fixed-seed EMA
  sampling.
- Component control: no change to forward schedule, `q_sample`, clean-image
  reconstruction, primary U-Net, EMA formula, checkpoint schema, or DDPM
  reverse equations.
- Data control: official CIFAR-10 Python archive, complete 50,000-image training
  split, deterministic `[-1,1]` transform, no augmentation.
- Sampling control: identical saved initial Gaussian tensors and identical
  reverse generator seed at every EMA checkpoint.
- Training control: one configuration and one real run; no sweep.

## Frozen model and diffusion configuration

| Field | Value |
|---|---|
| Model | existing `primary_unet_config()` |
| Parameters | `12,852,547` trainable |
| Base channels | `64` |
| Channel multipliers | `[1,2,2,4]` |
| Residual blocks per level | `2` |
| Attention resolutions | `[16]` |
| Time embedding dimension | `256` |
| Dropout | `0.1` |
| Prediction target | epsilon |
| Schedule | linear DDPM |
| Timesteps | `1000` |
| Beta start / end | `1e-4` / `2e-2` |
| Reverse variance | fixed DDPM posterior variance |
| Sampling | 1,000-step ancestral DDPM, clipped predicted `x_start` |

## Frozen training configuration

| Field | Value |
|---|---|
| Dataset | full CIFAR-10 training split, 50,000 images |
| Image shape / range | `[3,32,32]`, `[-1,1]` |
| Augmentation | none |
| Batch size | target `128` |
| Batch policy | deterministic epoch permutation, drop last |
| Workers | `4` |
| Optimizer | AdamW |
| Learning rate | `2e-4`, constant |
| Betas / epsilon | `(0.9,0.999)` / `1e-8` |
| Weight decay | `0` |
| EMA decay | `0.9999` |
| Gradient clipping | global L2 norm `1.0` |
| Precision | float32; no AMP |
| Model seed | `101` |
| Data-order seed | `103` |
| Training timestep/noise seed | `107` |
| Fixed initial-noise seeds | `1000..1015` |
| Fixed reverse-noise seed | `109` |
| Deterministic algorithms | enabled |
| Initial maximum | `50,000` steps |
| Recovery checkpoint cadence | every `5,000` real-run steps |
| Headline sample checkpoints | `10k`, `25k`, `50k` |

At batch 128, 50,000 steps process 6,400,000 images (`6.4 Mimg`) and
approximately 128 dataset equivalents. The corresponding milestone exposures
are 1.28 Mimg at 10k, 3.2 Mimg at 25k, and 6.4 Mimg at 50k.

Batch size 128 is frozen for try01. If and only if Gate A demonstrates an
out-of-memory failure, preserve try01 as failed and create try02 with a smaller
batch. Do not change the architecture, schedule, optimizer, learning rate, EMA,
or gradient clipping to address memory.

## One production optimization step

The order is invariant:

1. move one deterministic CIFAR-10 batch to the device;
2. sample one random code timestep per image;
3. sample one Gaussian epsilon tensor;
4. construct `x_t` with the validated `q_sample`;
5. predict epsilon with the primary U-Net;
6. compute mean squared error against sampled epsilon;
7. `optimizer.zero_grad(set_to_none=True)`;
8. `loss.backward()`;
9. reject missing or nonfinite gradients;
10. clip the global gradient norm to `1.0`;
11. `optimizer.step()`;
12. `ema.update(model)` after the optimizer changes the model;
13. log loss, pre/post-clip norms, throughput, elapsed time, and memory;
14. save due checkpoints without overwriting;
15. generate fixed-seed EMA samples only at configured evaluation points.

EMA never receives gradients and is never passed to the optimizer.

## Deterministic data and resume policy

The batch for global step `s` is derived from a deterministic epoch permutation
seeded by `data_seed + epoch`. Resume reconstructs the batch sampler from the
checkpoint global step rather than relying on opaque DataLoader iterator state.
The training timestep/noise generator and CUDA/global RNG state are stored by
the existing checkpoint schema. Dropout randomness is therefore restored with
the CUDA RNG state.

Gate B deliberately interrupts after 250 updates, stages the checkpoint,
destroys the process, then starts a new Slurm job that loads it and continues
to 500. Exact bitwise equivalence to an uninterrupted CUDA run is not required;
successful configuration-checked restoration, correct next step, finite
metrics, and continued progress are required.

## Compute and storage

- Execution: Hellbender GPU partition through Slurm only.
- GPU request: one available CUDA GPU with at least 32 GB.
- CPU/memory request: 8 CPUs and 64 GB host memory.
- Environment: persistent versioned environment under
  `~/data/diffusion-models-from-scratch/envs/`, created through Slurm.
- Source checkout: `~/projects/diffusion-models-from-scratch`, pulled to the
  exact published run commit before submission.
- Master dataset: `~/datasets/cifar10`; never copied into persistent outputs.
- Job-local caches and work: `$SLURM_TMPDIR` with all required cache exports.
- Persistent try outputs:
  `~/data/diffusion-models-from-scratch/exp004-full-cifar10-ddpm-training/try01/`.
- Local/GitHub: only selected lightweight summaries and figures after
  completion; bulky checkpoints remain on Hellbender.

Every training entrypoint refuses to run when `SLURM_JOB_ID` is absent.

## Preflight gates

### Infrastructure check

Submit `cluster/sbatch_checks.sh`. It must compile Python sources, import the
package, run CPU tests, and avoid dataset access and training.

### Gate A: 20-step real-data smoke

- One Slurm GPU job, full primary model, batch 128, official dataset.
- Exactly 20 finite loss records and gradient records.
- Model parameters change.
- EMA update count equals 20 and EMA differs from raw model.
- Write one structured step-20 checkpoint and a manifest.
- Record GPU identity, peak allocated/reserved memory, throughput, runtime,
  dataset identity, config hash, and run commit.
- Stop immediately on nonfinite loss/gradient or OOM.

Gate A does not sample and has no loss-reduction or image-quality threshold.

### Gate B: 500-step two-job preflight

1. Fresh job trains steps 1–250 and saves/stages step 250.
2. A new job reconstructs objects, validates configuration identity, restores
   step 250, and trains through step 500.
3. Save full 500-record loss/gradient history, step-500 checkpoint, manifest,
   throughput, runtime, and peak memory.
4. Copy EMA into an evaluation model and generate four fixed-seed images with
   the validated 1,000-step DDPM sampler.
5. Repeat the checkpoint sampling with the same fixed initial tensors and
   reverse seed; tensor hashes must match exactly.

Gate B passes when all 500 records are present and finite, global step and EMA
count equal 500, resume is configuration-checked, sampling executes and is
reproducible, and artifacts validate. Sample quality at 500 steps is explicitly
not a criterion.

Do not start the long run if Gate B fails.

## Gate C: first real run

Launch one run from step zero after Gate B passes. The initial authorized
ceiling is 50,000 steps. Save recovery checkpoints every 5,000 steps. At 10k,
25k, and 50k:

1. preserve the complete raw-model/EMA/optimizer/RNG checkpoint;
2. generate the same 16 fixed-initial-noise EMA samples;
3. use fixed reverse-noise seed 109;
4. save the model-space sample tensor and inverse-normalized grid;
5. preserve a loss-curve snapshot;
6. record wall time, throughput, peak memory, and checkpoint hash.

The headline progression figure uses the identical 16 rows/seeds at all three
checkpoints. No sample or checkpoint may be overwritten. The optional 100k
extension is not part of this plan and requires a new reviewed decision after
the frozen 50k result is inspected.

## Frozen evaluation rules

- Structural stability: all loss, pre-clip gradient norm, post-clip gradient
  norm, parameters, EMA tensors, and samples remain finite.
- Loss decrease: mean loss over steps 49,001–50,000 must be below 90% of the
  mean loss over steps 1–1,000. This threshold is frozen before execution and
  is intentionally modest.
- Sample progression: inspect the same fixed-seed rows at 10k, 25k, and 50k.
  Report observations without cherry-picking or substituting seeds.
- Recognizability: the allowed end-to-end conclusion requires visible coherent
  CIFAR-like object structure in the frozen 50k grid. If samples remain noise-
  like or amorphous, report stable training separately and mark the full-system
  objective inconclusive or failed rather than changing the criterion.
- No FID/KID is computed in this milestone.

## Expected outputs

Gate A:

- 20 JSONL metric records;
- one step-20 checkpoint;
- one segment summary and one run manifest;
- no sample figure.

Gate B:

- 500 total JSONL metric records;
- checkpoints at steps 250 and 500;
- two segment summaries/manifests;
- one four-image fixed-seed sample tensor/grid at step 500;
- one reproducibility comparison record.

Gate C:

- 50,000 JSONL metric records;
- checkpoints at 5k multiples through 50k;
- sample tensors/grids at 10k, 25k, and 50k;
- loss curves at the same milestones;
- segment/run manifest with hashes and system identity;
- one fixed-seed progression figure after validation.

## Failure and stop conditions

- Stop on nonfinite loss, gradient, parameter, EMA state, or sample.
- Stop on checkpoint/configuration mismatch or failed resume.
- Stop if source commit differs from the submitted run commit or the execution
  checkout is dirty.
- Stop if dataset identity or count differs from the recorded full training
  split.
- Stop Gate A on OOM; preserve the failed try before any batch adjustment.
- Do not stop solely because 500-step or early real-run samples look poor.
- Do not alter seeds, schedule, architecture, optimizer, or evaluation rules
  after observing results within a try.
- Do not sweep hyperparameters, add AMP/distributed training, implement DDIM,
  or add FID/KID.
- Stop after the first successful 50k trained model and frozen progression.

## Unit tests for new orchestration

Add CPU-only tests for:

1. deterministic step-indexed batch ordering and resume position;
2. one production training step with finite metrics and parameter change;
3. explicit `optimizer.step()` before `ema.update(model)`;
4. EMA update count increments exactly once per successful step;
5. checkpoint cadence and headline sampling cadence;
6. resume wiring with configuration guard;
7. fixed initial sample tensor preservation and seed identity;
8. Slurm runtime refusal at the entrypoint boundary.

Do not duplicate mathematical tests and do not load CIFAR-10 in normal CI.

## Documentation plan

Create `docs/understanding/full_training.md` and `Reports/milestone_06.md`.
Update the `EXP004` overview, hypothesis, try metadata/report, project state,
experiment index, timeline, README, and project specification as evidence
becomes available. The report must distinguish plan, implementation, submitted
jobs, observations, interpretation, and unverified claims.

## Unresolved items

- Actual GPU model and usable memory are assigned by Slurm and are nonblocking
  until Gate A records them.
- Exact production run commit is created after implementation validation and
  is nonblocking; every job will record and enforce it.
- Real-run runtime is unknown until Gate A/B throughput measurements. This is
  nonblocking but must be estimated before Gate C submission.

