# DDPM Reverse Sampling Plan

## Status and authority

This is the authoritative implementation and execution plan for Milestone 5
and `EXP003/try01`. It was frozen on 2026-08-07 before reverse-sampling code
was implemented or the 1,000-step smoke trajectory was run.

Milestone 4 is closed at commit
`2f13b3f785b6adfd815199b5a6274dd6b5245cee`. Push-triggered GitHub Actions run
`31152749677` passed on `main` with 41 tests plus Ruff lint and formatting. The
worktree was clean before the `EXP003` scaffold was created.

## Research question

Does a direct PyTorch implementation of the DDPM posterior and ancestral
reverse chain satisfy the derived Gaussian equations, handle the deterministic
final step correctly, and execute all 1,000 reverse updates reproducibly with
the existing time-conditioned U-Net interface?

## Motivation and existing evidence

Milestone 1 validates the forward marginal and exact-noise reconstruction.
Milestone 2 validates the time-conditioned U-Net shape contract. Milestone 3
establishes epsilon-prediction learnability on a fixed tiny dataset, and
Milestone 4 provides model-independent EMA weights and exact checkpoint resume.
The missing component is the reverse transition that converts a model noise
prediction into one ancestral DDPM step and composes those steps into a chain.

## Hypothesis and interpretation

If the posterior coefficients are derived and indexed correctly, then they
will match an independent hand calculation, `p_sample` will equal the posterior
mean at code timestep zero, supplied reverse noise will make every other step
deterministic, and a seeded 1,000-step chain will be finite, shape-preserving,
and reproducible.

Passing supports structural correctness of DDPM ancestral sampling. Because
the smoke run uses randomly initialized weights, it cannot support any claim
about image quality, learned generation, FID, or CIFAR-10 generalization.

## Baseline, intervention, and controls

- Baseline: Milestone 4 commit `2f13b3f` with forward diffusion, U-Net, EMA,
  and checkpointing but no reverse sampler.
- Intervention: posterior coefficient calculation, model-mean conversion,
  one-step ancestral sampling, and a complete reverse loop.
- Mathematical control: a two-step float64 schedule calculated independently.
- Noise control: explicitly supplied reverse noise for deterministic one-step
  comparisons.
- Final-step control: two different noise tensors must produce the same result
  at code timestep zero.
- Seed control: equal generator seeds must produce equal chains; a different
  seed must produce a different chain.
- Interface control: an EMA-weighted model copy is passed through the ordinary
  model argument; the sampler has no EMA-specific branch.

## Mathematical definition

For mathematical timesteps `t in {1,...,T}`, define

```math
\tilde\beta_t = \beta_t\frac{1-\bar\alpha_{t-1}}
{1-\bar\alpha_t},
```

and

```math
\tilde\mu_t(x_t,x_0) =
\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0
+\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}
{1-\bar\alpha_t}x_t.
```

The epsilon-prediction model reconstructs

```math
\hat x_0 = \frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta(x_t,t)}
{\sqrt{\bar\alpha_t}}.
```

The implementation uses code index `i=t-1`; `alpha_bar_prev[0]=1`. At code
index zero, `posterior_variance=0`, so `p_sample` returns the posterior mean
without drawing or adding noise. For `i>0`, it returns

```math
x_{t-1}=\tilde\mu_t+\sqrt{\tilde\beta_t}z,
\qquad z\sim\mathcal N(0,I).
```

## Frozen scientific invariants

- The existing U-Net and validated forward equations remain unchanged.
- Images have NCHW layout; the smoke shape is `[4,3,32,32]` in `float32`.
- The schedule remains linear with `T=1000`, beta `1e-4` to `2e-2`.
- The model predicts epsilon, not `x_start`, score, or velocity.
- The estimated `x_start` is clipped to model space `[-1,1]` by default before
  posterior-mean calculation; the API exposes a boolean switch for explicit
  diagnostic use.
- Visualization alone applies inverse normalization and display-boundary
  clamping.
- The reverse variance is the fixed posterior variance `beta_tilde`.
- The sampler accepts an ordinary callable model. EMA use occurs by copying
  EMA state into a model before calling the sampler.
- Sampling runs under no-gradient evaluation and never updates weights.
- No dataset download, training, checkpoint quality evaluation, DDIM, GPU,
  cluster job, or image-quality metric is permitted.

## Public API and trajectory convention

- `q_posterior_mean_variance(x_start, x_t, timesteps, schedule)` returns a
  `PosteriorOutput(mean, variance)`.
- `p_mean_variance(model, x_t, timesteps, schedule, clip_x_start=True)` returns
  the posterior mean/variance plus predicted epsilon and `x_start`.
- `p_sample(...)` returns one reverse sample and accepts either a generator or
  a supplied reverse-noise tensor.
- `p_sample_loop(...)` returns `SamplingResult(sample, trajectory)`.

The loop starts from `x_T`. Its trajectory dictionary uses the number of
remaining transitions as the state label: key `T` is the initial Gaussian,
and after processing code timestep `i`, the resulting state is keyed by `i`.
Thus the frozen requested keys `[1000,750,500,250,100,0]` mean
`x_T,x_750,x_500,x_250,x_100,x_0`. The model must be called exactly `T` times.

## Frozen smoke configuration

| Field | Value |
|---|---|
| Model | existing 491,107-parameter smoke U-Net |
| Weight source | EMA initialized as an exact copy of the random model |
| Model seed | `71` |
| Sampling generator seed | `73` |
| Batch and shape | `4`, `[4,3,32,32]` |
| Device / dtype | CPU / `float32` |
| PyTorch threads | `1` |
| Deterministic algorithms | enabled |
| Schedule | linear, `T=1000`, beta `1e-4` to `2e-2` |
| Clip predicted `x_start` | `true` |
| Trajectory keys | `[1000,750,500,250,100,0]` |
| Baseline commit | `2f13b3f785b6adfd815199b5a6274dd6b5245cee` |

Exact command:

```bash
.venv/bin/python scripts/run_sampling_smoke.py \
  --config configs/sampling_smoke.json
```

## Test gates

### Gate A: posterior mathematics

- Match independent two-step float64 posterior variance and both mean
  coefficients.
- Verify finite, nonnegative variance, code-timestep-zero safety, and NCHW
  broadcasting.

### Gate B: one reverse step

- Use a dummy epsilon model with known output.
- Match a manual mean and supplied-noise sample at a nonzero timestep.
- Prove different supplied noise cannot change the timestep-zero result.
- Preserve shape and reject invalid timestep/noise inputs clearly.

### Gate C: complete loop

- Verify exactly `T` model calls and output shape.
- Verify finite output for a random smoke U-Net.
- Verify same seeds reproduce exactly and different seeds change output.
- Verify an EMA-populated model can be passed through the same API.
- Execute one full `T=1000` CPU smoke chain and save its requested trajectory.

Normal CI uses short synthetic schedules and no dataset. It must not assert
that random-model images look good.

## Expected artifacts and counts

- `configs/sampling_smoke.json`: one frozen configuration.
- `EXP003/try01/results/sampling_summary.json`: one structured result.
- `EXP003/try01/results/final_sample.pt`: one final model-space tensor.
- `EXP003/try01/figures/random_model_ddpm_trajectory.png`: one six-column
  debug trajectory.
- Exactly six recorded trajectory tensors and exactly 1,000 model calls.
- Existing completed experiment outputs remain unchanged.

## Failure and stop conditions

- Stop on a mismatch with independent posterior arithmetic.
- Stop if variance is negative/nonfinite or the final step adds noise.
- Stop if shapes/devices/dtypes drift or any sample becomes nonfinite.
- Stop if the loop does not call the model exactly 1,000 times.
- Stop if equal seeds differ or the different-seed control is identical.
- Do not change the schedule, seeds, clipping rule, or acceptance criteria after
  observing results within this try.
- Do not train a full model or interpret random-model visuals as quality.

## Acceptance criteria

- Every Gate A/B/C test above passes.
- The full 1,000-step smoke command exits normally on CPU.
- Summary reports finite `[4,3,32,32]` output, six requested trajectory keys,
  and exactly 1,000 model calls.
- The trajectory PNG is visually inspected for correct layout and labels.
- Pytest, Ruff lint, Ruff formatting, and `git diff --check` pass.
- The implementation, derivation, limitations, hashes, and exact command are
  documented durably.

## Documentation plan

Create `docs/ddpm_reverse_process.md`, `docs/understanding/sampling.md`, and
`Reports/milestone_05.md`; update the `EXP003` record, project specification,
README, state, current context, thinking, experiment index, and timeline after
validated execution.

