#!/usr/bin/env bash
#SBATCH --job-name=spectral_noise_checks
#SBATCH --partition=logical_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail
job_scratch=${SLURM_TMPDIR:-/tmp/${USER}/spectral-noise-checks-${SLURM_JOB_ID}}
export TMPDIR="$job_scratch/tmp"
export XDG_CACHE_HOME="$job_scratch/.cache"
export TORCH_HOME="$job_scratch/torch"
export MPLCONFIGDIR="$job_scratch/matplotlib"
export PIP_CACHE_DIR="$job_scratch/pip-cache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" "$PIP_CACHE_DIR"
repository=/home/xggh8/projects/diffusion-models-from-scratch-noise
environment_root=/home/xggh8/data/diffusion-models-from-scratch/envs/torch-2.13.0
: "${RUN_COMMIT:?RUN_COMMIT is required}"
[[ "$(git -C "$repository" rev-parse HEAD)" == "$RUN_COMMIT" ]]
[[ -z "$(git -C "$repository" status --porcelain)" ]]
cd "$repository"
"$environment_root/bin/python" -m compileall -q src scripts
"$environment_root/bin/python" -m pytest -q
"$environment_root/bin/ruff" check .
"$environment_root/bin/ruff" format --check .
