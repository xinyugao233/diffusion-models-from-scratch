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
```

The 50k command is submitted only after Gate A and Gate B artifacts validate.
All caches and active work remain in `$SLURM_TMPDIR` when the cluster provides
it; otherwise the scripts use a job-specific directory under `/tmp`. Only
required logs, checkpoints, metrics, manifests, and figures are staged to
`~/data`.
