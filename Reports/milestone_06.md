# Milestone 06 Report: First Full CIFAR-10 DDPM Training

## Outcome

`COMPLETE`. The unchanged 12,852,547-parameter DDPM trained for 50,000 steps
on the full CIFAR-10 training split. All frozen infrastructure, resume,
stability, loss, cadence, and qualitative sample gates passed.

![Fixed-seed EMA progression](../Experiments/exp004-full-cifar-10-ddpm-training/try01/figures/full/fixed_seed_progression_010000_025000_050000.png)

## Exact identity

- Run commit: `7ad4524fbbeb61fc78a024ee74e2638644012774`
- GitHub Actions run: `31155794394`, successful
- Configuration SHA-256:
  `8d4480ef52c2d299320cc374ae893b07a16690e091c88f303d1b9c8bf0140487`
- CIFAR-10 archive MD5: `c58f30108f718f92721af3b95e74349a`
- Final job: `15943245`, exit `0:0`, NVIDIA H100 NVL
- Environment: Python 3.13.11, PyTorch 2.13.0+cu130, CUDA 13.0

## Gate results

| Gate | Requirement | Evidence | Status |
|---|---|---|---|
| Infrastructure | Slurm setup and dataset-free checks | Jobs `15914382` and `15914409`; 61 tests, lint, format | PASS |
| Gate A | 20 finite updates, parameter/EMA change, checkpoint | Job `15914422`; 20 records, EMA count 20 | PASS |
| Gate B | 250+250 new-process resume and exact repeated sample | Jobs `15914482`, `15938195`; 500 records, matching tensor hash | PASS |
| Gate C records | 50,000 unique contiguous finite records | Raw JSONL SHA-256 `1fbd7e...4896f` | PASS |
| Gate C loss | Last-1k mean below 90% of first-1k mean | `0.0305275 / 0.0655694 = 0.465576` | PASS |
| Checkpoints | Every 5k through 50k | Ten immutable 205,996,222-byte checkpoints | PASS |
| Sampling | Fixed-seed grids at 10k, 25k, 50k | Three tensors/grids and visually inspected progression | PASS |
| Recognizability | Coherent CIFAR-like structure at 50k | Multiple animal- and vehicle-like forms in frozen grid | PASS |

## Quantitative result

- First loss: `1.1303887367`
- Final loss: `0.0336870030`
- First 1,000-step mean: `0.0655694469`
- Last 1,000-step mean: `0.0305275310`
- Late/early ratio: `0.4655755453` (frozen maximum `0.9`)
- Runtime: `3,108.897 s` training time; Slurm elapsed `00:52:02`
- Throughput: `16.083` steps/s, `2,058.608` images/s
- Peak allocated/reserved device memory: 3.035 / 3.506 GB
- Final checkpoint SHA-256:
  `f6b64f7028432ac1690029564610918e68e0c45b5188a97961863eadd79b17b9`

## Qualitative result

The same preregistered 16 initial-noise seeds and reverse seed were used at all
three checkpoints. The 10k grid is dominated by coarse texture; the 25k grid
develops color-separated silhouettes; the 50k grid contains coherent
foreground/background layouts and recognizable CIFAR-like animals and
vehicles. The fixed progression, not selected individual seeds, determines the
qualitative pass.

## Preserved failures

Setup job `15914281` failed before environment creation because Hellbender did
not export `SLURM_TMPDIR`. Commit `7ad4524` added a job-local `/tmp` fallback.
The first Gate B attempt, job `15914462`, failed before step 1 after the
ambiguous GPU request landed on a V100 unsupported by the pinned CUDA wheel.
Its log is preserved under `gate_b_failed_job_15914462`; successful GPU work
was constrained to H100 without changing any scientific setting.

## Valid conclusion and limitations

The complete DDPM implementation trains stably on CIFAR-10, restores correctly
across a new process, and produces increasingly structured, recognizable EMA
samples with the validated 1,000-step ancestral sampler. This milestone does
not evaluate FID/KID, likelihood, held-out generalization, DDIM, alternative
architectures or schedules, class conditioning, AMP, or distributed training.

## Evidence

- Authoritative plan: `docs/plans/full_cifar10_training.md`
- Experiment report:
  `Experiments/exp004-full-cifar-10-ddpm-training/try01/report.md`
- Independent metrics audit:
  `Experiments/exp004-full-cifar-10-ddpm-training/try01/results/full/metrics_summary.json`
- Remote immutable outputs:
  `/home/xggh8/data/diffusion-models-from-scratch/exp004-full-cifar10-ddpm-training/try01/`

## Stop decision

The preregistered 50k objective is satisfied. No 100k extension was launched,
and DDIM remains a separate future milestone requiring a reviewed plan.
