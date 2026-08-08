# EXP004 Try 01: Full CIFAR-10 DDPM Training

## Status

`COMPLETED` — all staged gates and the frozen 50,000-step run completed, and
their outputs were independently validated.

## Goal and hypothesis tested

Test whether the unchanged primary U-Net, linear DDPM, epsilon objective,
AdamW, EMA, checkpointing, and ancestral sampler train stably together on all
50,000 CIFAR-10 training images and produce recognizable fixed-seed EMA
samples. The authoritative preregistration is
`docs/plans/full_cifar10_training.md`.

## Frozen configuration and provenance

- Run commit: `7ad4524fbbeb61fc78a024ee74e2638644012774`
- GitHub Actions: `31155794394`, successful
- Config SHA-256:
  `8d4480ef52c2d299320cc374ae893b07a16690e091c88f303d1b9c8bf0140487`
- CIFAR-10 archive MD5: `c58f30108f718f92721af3b95e74349a`
- CIFAR-10 archive SHA-256:
  `6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce`
- Model: unchanged 12,852,547-parameter primary U-Net
- Software: Python 3.13.11, PyTorch 2.13.0+cu130, CUDA 13.0, cuDNN 92000
- Successful GPU jobs: NVIDIA H100 NVL

## Execution

| Job | Purpose | State | Evidence |
|---|---|---|---|
| `15914281` | Initial environment setup | FAILED | No environment was created; `SLURM_TMPDIR` was unset. |
| `15914382` | Repaired environment setup | COMPLETED | Installed pinned environment and wrote `READY`. |
| `15914409` | Dataset-free Slurm checks | COMPLETED | 61 tests, Ruff lint, and format check passed. |
| `15914422` | Gate A, 20 steps | COMPLETED | 20 finite records and step-20 checkpoint. |
| `15914462` | Gate B first attempt | FAILED BEFORE STEP 1 | Scheduler assigned a V100 unsupported by the CUDA 13 wheel. |
| `15914482` | Gate B, steps 1–250 | COMPLETED | Step-250 checkpoint staged on H100. |
| `15938195` | Gate B, new-process steps 251–500 | COMPLETED | Resume validated; exact repeated sample hash matched. |
| `15943245` | Gate C, fresh steps 1–50,000 | COMPLETED | Exit `0:0`, elapsed `00:52:02`, node `g033`. |

The failed Gate B log is preserved remotely under
`gate_b_failed_job_15914462/`. The successful Gate B was submitted with an
H100 resource constraint; no scientific configuration changed.

## Verification and results

### Gate A

- Exactly 20 contiguous finite metric records.
- Loss `1.1303887367 -> 0.1686984003`.
- Parameters changed; EMA differed from the model and reached 20 updates.
- Peak allocated/reserved CUDA memory: 3,035,344,896 / 3,502,243,840 bytes.
- Gate status: PASS.

### Gate B

- Exactly 500 contiguous finite metric records across two Slurm processes.
- The second process restored the config, dataset identity, global step, model,
  optimizer, EMA, and RNG state from the step-250 checkpoint.
- EMA update count reached 500; final loss was `0.0555764064`.
- First/last 100-step mean losses were `0.1925850528` and `0.0499065569`.
- Repeated four-image sampling produced the identical tensor SHA-256
  `09eaffc63f22eade4bce464eb7537f7f84254db764f7457b30930aba1376cc20`.
- The 500-step samples were noise-like, as expected; quality was not a gate.
- Gate status: PASS.

### Gate C

- Exactly 50,000 unique, contiguous JSONL records; every numeric value was
  finite.
- Mean loss over steps 1–1,000: `0.0655694469`.
- Mean loss over steps 49,001–50,000: `0.0305275310`.
- Late/early ratio: `0.4655755453`, below the frozen maximum `0.9`.
- Final loss: `0.0336870030`; observed range:
  `[0.0131198727, 1.1303887367]`.
- EMA updates: 50,000.
- Ten immutable checkpoints exist at every 5k step through 50k.
- Three sample tensors, inverse-normalized grids, and loss curves exist at
  10k, 25k, and 50k.
- Runtime: 3,108.897 training seconds; 16.083 steps/s and 2,058.608 images/s.
- Peak allocated/reserved CUDA memory: 3,035,344,896 / 3,506,438,144 bytes.
- Raw metrics SHA-256:
  `1fbd7e08897f21e0c2fe056bef465e2d11e2a42d106907f04920ec2f0de4896f`.
- Final checkpoint SHA-256:
  `f6b64f7028432ac1690029564610918e68e0c45b5188a97961863eadd79b17b9`.
- Gate status: PASS.

## Figure inspection

The frozen seed rows were inspected without substitution. At 10k the images
are mostly coarse warm textures. At 25k they show clearer color separation and
incipient silhouettes. At 50k multiple samples have coherent CIFAR-like
foreground/background composition and recognizable animal- or vehicle-like
forms. This passes the preregistered qualitative recognizability criterion.

The loss curve was also inspected. It shows a sharp initial decrease followed
by a finite noisy plateau; the exact windowed test above, not the plot shape,
determines the quantitative pass.

## Failures and limitations

The two infrastructure failures are informative and preserved: Hellbender did
not export `SLURM_TMPDIR`, and an unconstrained GPU request could land on a V100
unsupported by the pinned CUDA wheel. Neither failure executed an optimization
step or changed the frozen training trajectory.

No FID/KID, held-out likelihood, DDIM, architecture comparison,
hyperparameter sweep, class conditioning, AMP, distributed training, or second
dataset was evaluated. Visual recognizability is a preregistered qualitative
gate, not a competitive quality metric.

## Interpretation

The evidence supports the limited planned conclusion: the complete DDPM
implementation trains stably on CIFAR-10, survives a checked new-process
resume, and its EMA checkpoints progress to recognizable CIFAR-like samples
under the validated 1,000-step ancestral sampler. It does not support a
state-of-the-art, comparative, generalization, or DDIM claim.

## Exact evidence paths

- Local metrics audit: `results/full/metrics_summary.json`
- Local progression: `figures/full/fixed_seed_progression_010000_025000_050000.png`
- Local final summary: `results/full/segment_summary_000000_050000.json`
- Preserved failure logs: `results/failures/setup-15914281.out` and
  `results/failures/gate-b-250-15914462.out`
- Remote raw outputs:
  `/home/xggh8/data/diffusion-models-from-scratch/exp004-full-cifar10-ddpm-training/try01/`
- Remote full log:
  `/home/xggh8/data/diffusion-models-from-scratch/slurm-logs/full-50000-15943245.out`

## Next step

Close Milestone 6 at 50k. Do not extend training or begin DDIM without a new
reviewed plan.
