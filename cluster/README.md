# Hellbender Slurm Execution

Milestone 6 runs only through Slurm. Source is authored locally, published to
GitHub, and pulled into `/home/xggh8/projects/diffusion-models-from-scratch` at
the exact run commit.

Submission commands are intentionally explicit:

```bash
run_commit=$(git rev-parse HEAD)
log_root=/home/xggh8/data/diffusion-models-from-scratch/slurm-logs
mkdir -p "$log_root"

sbatch --output="$log_root/setup-%j.out" cluster/sbatch_setup_env.sh
sbatch --export=ALL,RUN_COMMIT="$run_commit" \
  --output="$log_root/checks-%j.out" cluster/sbatch_checks.sh

sbatch --export=ALL,RUN_COMMIT="$run_commit" \
  --output="$log_root/gate-a-%j.out" \
  cluster/sbatch_train.sh gate_a 20

sbatch --export=ALL,RUN_COMMIT="$run_commit" \
  --output="$log_root/gate-b-250-%j.out" \
  cluster/sbatch_train.sh gate_b 250

sbatch --export=ALL,RUN_COMMIT="$run_commit" \
  --output="$log_root/gate-b-500-%j.out" \
  cluster/sbatch_train.sh gate_b 500 checkpoint_step_000250.pt

sbatch --export=ALL,RUN_COMMIT="$run_commit" \
  --output="$log_root/full-50000-%j.out" \
  cluster/sbatch_train.sh full 50000
```

The training script requests the H100 type validated with the pinned CUDA 13
wheel. The 50k command is submitted only after Gate A and Gate B artifacts
validate. Do not rerun an existing completed stage; create a new numbered try.
All caches and active work remain in `$SLURM_TMPDIR` when the cluster provides
it; otherwise the scripts use a job-specific directory under `/tmp`. Only
required logs, checkpoints, metrics, manifests, and figures are staged to
`~/data`.

## Milestone 7 sampler comparison

The deterministic DDIM comparison reuses the completed 50k EMA checkpoint. It
does not train or download a dataset:

```bash
run_commit=$(git rev-parse HEAD)
sbatch --export=ALL,RUN_COMMIT="$run_commit" \
  --output="$log_root/ddim-comparison-%j.out" \
  cluster/sbatch_ddim_comparison.sh
```

The launcher refuses an existing `EXP005/try01`, verifies the exact clean run
commit, copies the checkpoint to job-local scratch, and stages the small result
bundle to `~/data/diffusion-models-from-scratch/exp005-ddim-comparison/try01`.

## Spectral-boundary 50K spectrum gate

Before intervention training, recompute the exact E006 radial statistic over
all 50,000 CIFAR-10 training images. The job refuses non-Slurm execution and
existing attempt directories:

```bash
run_commit=$(git rev-parse HEAD)
sbatch --export=ALL,RUN_COMMIT="$run_commit",ATTEMPT=try01 \
  --output="$log_root/spectrum-50k-%j.out" \
  cluster/sbatch_spectrum_50k.sh
```

Results are staged to
`~/data/diffusion-models-from-scratch/spectrum-50k/try01`. The predeclared
materiality rule uses the 50K spectrum for intervention training when the
maximum absolute per-shell relative difference from E006 exceeds 5%, more than
5% of DDPM timesteps change crossing shell, or any crossing moves by more than
one shell. Otherwise the E006 first-1K spectrum remains frozen.
