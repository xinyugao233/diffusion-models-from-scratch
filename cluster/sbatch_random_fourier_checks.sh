#!/usr/bin/env bash
#SBATCH --job-name=fourier_pair_checks
#SBATCH --partition=logical_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail

job_scratch=${SLURM_TMPDIR:-/tmp/${USER}/fourier-pair-checks-${SLURM_JOB_ID}}
export TMPDIR="$job_scratch/tmp"
export XDG_CACHE_HOME="$job_scratch/.cache"
export TORCH_HOME="$job_scratch/torch_cache"
export MPLCONFIGDIR="$job_scratch/matplotlib"
export WANDB_DIR="$job_scratch/wandb"
export HF_HOME="$job_scratch/hf"
export TRANSFORMERS_CACHE="$job_scratch/hf/transformers"
export HF_DATASETS_CACHE="$job_scratch/hf/datasets"
export PIP_CACHE_DIR="$job_scratch/pip-cache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" \
  "$WANDB_DIR" "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" \
  "$PIP_CACHE_DIR"

repository=/home/xggh8/projects/diffusion-models-from-scratch
environment_root=/home/xggh8/data/diffusion-models-from-scratch/envs/torch-2.13.0
if [[ -z "${RUN_COMMIT:-}" ]]; then
  echo "RUN_COMMIT is required" >&2
  exit 1
fi
if [[ "$(git -C "$repository" rev-parse HEAD)" != "$RUN_COMMIT" ]]; then
  echo "Repository commit differs from RUN_COMMIT" >&2
  exit 1
fi
if [[ -n "$(git -C "$repository" status --porcelain)" ]]; then
  echo "Repository checkout is dirty" >&2
  exit 1
fi
cd "$repository"
"$environment_root/bin/python" -m compileall -q src scripts
"$environment_root/bin/python" -c 'import diffusion_models.fourier'
"$environment_root/bin/python" -m pytest -q
"$environment_root/bin/ruff" check .
"$environment_root/bin/ruff" format --check .
