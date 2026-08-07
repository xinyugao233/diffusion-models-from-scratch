# Milestone 03 Report: Epsilon-Prediction Learnability

## Objective

Establish that the validated smoke U-Net can optimize the explicit DDPM
epsilon-prediction objective on a fixed synthetic problem and on repeated
stochastic noising of 16 fixed CIFAR-10 training images.

## Baseline and scope

- Baseline commit: `26e735e597001251d92075d5bf02fb5649851800`
- Executed code state: uncommitted, with exact source/config hashes preserved in
  `EXP001/try01` summaries
- Device: local Apple arm64 CPU, one PyTorch thread
- Model: unchanged 491,107-parameter smoke U-Net
- Schedule: unchanged linear 1,000-step schedule, beta `1e-4` to `2e-2`
- Excluded: full-dataset training, GPU/cluster work, EMA, checkpointing,
  reverse sampling, DDIM, AMP, and distributed training

## Implementation

`src/diffusion_models/training.py` now makes the noise coupling explicit,
computes scalar epsilon MSE, performs checked optimizer updates, calculates and
clips global gradient norms, rejects nonfinite values, and logs immutable JSONL
scalars. Two frozen scripts execute the synthetic and fixed-16 gates. The
existing U-Net and forward-diffusion implementation were not changed.

## Verification before execution

```text
.venv/bin/python -m pytest -q
  -> 31 passed
.venv/bin/ruff check .
  -> All checks passed
.venv/bin/ruff format --check .
  -> 36 files already formatted
git diff --check
  -> passed with no output
```

Tests remain synthetic-only and perform no dataset access.

## Gate A: fixed synthetic batch

```text
Command: .venv/bin/python scripts/run_synthetic_optimization.py \
         --config configs/synthetic_optimization.json
Steps: 40
Batch: 4
Duration: 1.7245 seconds
First logged loss: 1.1447415
Final evaluated loss: 0.0053172
Early five-step mean: 0.7734421
Late five-step mean: 0.0082524
Late/early ratio: 0.0106697
Relative windowed decrease: 98.9330%
Initial/final gradient norm: 1.531378 / 0.059493
Parameter changed: yes
Acceptance maximum ratio: 0.80
Result: PASS
```

The smoke model intentionally has no attention. A separate two-update
diagnostic on the validated `SelfAttention2d` found zero QKV gradient on the
first backward pass, a nonzero output-projection gradient, and a nonzero QKV
gradient on the second backward pass after the output projection changed.

## Gate B: fixed 16-image CIFAR-10 overfit

```text
Command: .venv/bin/python scripts/run_overfit_16.py \
         --config configs/overfit_16.json --download
Steps / subset epochs: 500 / 500
Images processed: 8,000 (8 kimg, 0.008 mimg)
Batch: 16; every fixed example appears once per update
Duration: 102.9107 seconds
Initial loss: 1.0944936
Final logged loss: 0.0526395
Early 50-step mean: 0.2201840
Late 50-step mean: 0.0438726
Late/early ratio: 0.1992544
Relative windowed decrease: 80.0746%
Gradient norm min/mean/max: 0.049766 / 0.122540 / 1.709329
Initial/final gradient norm: 1.709329 / 0.092026
NaN or instability: none observed
Acceptance maximum ratio: 0.80
Result: PASS
```

The initial sandboxed download attempt failed before dataset access because DNS
was blocked. An approved official download then progressed too slowly and was
interrupted at 2.6%. The complete official CIFAR-10 archive and extracted files
already used by a sibling research project were staged into this repository's
ignored `data/` directory. Source and destination archive MD5 values both
matched torchvision's official value `c58f30108f718f92721af3b95e74349a`.
The unchanged command then ran with `--download`, allowing torchvision to
verify the cache before manifest construction and optimization.

## Frozen subset

| Index | Label | Class |
|---:|---:|---|
| 0 | 6 | frog |
| 1 | 9 | truck |
| 2 | 9 | truck |
| 3 | 4 | deer |
| 4 | 1 | automobile |
| 5 | 1 | automobile |
| 6 | 2 | bird |
| 7 | 7 | horse |
| 8 | 8 | ship |
| 9 | 3 | cat |
| 10 | 4 | deer |
| 11 | 7 | horse |
| 12 | 7 | horse |
| 13 | 2 | bird |
| 14 | 9 | truck |
| 15 | 9 | truck |

The durable manifest records exactly 16 unique indices, no augmentation, class
names, and observed normalized range `[-1,1]`.

## Evidence validation

- Synthetic JSONL: exactly 40 sequential finite records.
- CIFAR JSONL and CSV: exactly 500 mutually consistent sequential finite
  records.
- Recomputed early/late means match the saved summary within `1e-12`.
- The visually inspected loss curve is a valid 1,200×700 RGB PNG and shows the
  same rapid decrease and later stochastic fluctuations as the raw log.
- No checkpoint was created or evaluated; the evidence is the complete training
  loss record from the final completed run.
- A PDF report was not generated because this milestone explicitly requested
  Markdown/code artifacts and a loss figure, and the experiment has no
  checkpoint or sampling evaluation to present. This Markdown report and raw
  records are canonical.

## Scientific conclusion

The evidence supports the preregistered claim that the complete CIFAR-10
epsilon-prediction pipeline can learn a fixed small dataset. It does not show
that the model generates images, generalizes beyond the subset, trains stably
on all CIFAR-10, or achieves useful generative quality.

## Evidence paths

- Plan: `docs/plans/training_and_overfit.md`
- Manifest: `configs/cifar10_overfit_16_manifest.json`
- Synthetic summary: `Experiments/exp001-epsilon-prediction-learnability/try01/results/synthetic_summary.json`
- Overfit summary: `Experiments/exp001-epsilon-prediction-learnability/try01/results/overfit_summary.json`
- Raw loss CSV: `Experiments/exp001-epsilon-prediction-learnability/try01/results/overfit_loss.csv`
- Loss curve: `Experiments/exp001-epsilon-prediction-learnability/try01/figures/overfit_loss_curve.png`
