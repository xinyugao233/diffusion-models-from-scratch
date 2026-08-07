#!/usr/bin/env bash
#SBATCH --job-name=ddpm_checks
#SBATCH --partition=logical_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail

export TMPDIR="$SLURM_TMPDIR/tmp"
export XDG_CACHE_HOME="$SLURM_TMPDIR/.cache"
export TORCH_HOME="$SLURM_TMPDIR/torch_cache"
export MPLCONFIGDIR="$SLURM_TMPDIR/matplotlib"
export WANDB_DIR="$SLURM_TMPDIR/wandb"
export HF_HOME="$SLURM_TMPDIR/hf"
export TRANSFORMERS_CACHE="$SLURM_TMPDIR/hf/transformers"
export HF_DATASETS_CACHE="$SLURM_TMPDIR/hf/datasets"
export PIP_CACHE_DIR="$SLURM_TMPDIR/pip-cache"

mkdir -p \
  "$TMPDIR" \
  "$XDG_CACHE_HOME" \
  "$TORCH_HOME" \
  "$MPLCONFIGDIR" \
  "$WANDB_DIR" \
  "$HF_HOME" \
  "$TRANSFORMERS_CACHE" \
  "$HF_DATASETS_CACHE" \
  "$PIP_CACHE_DIR"

repository=/home/xggh8/projects/diffusion-models-from-scratch
environment_root=/home/xggh8/data/diffusion-models-from-scratch/envs/torch-2.13.0

if [[ ! -f "$environment_root/READY" ]]; then
  echo "Missing ready environment: $environment_root" >&2
  exit 1
fi
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
"$environment_root/bin/python" -c \
  'import diffusion_models; print(diffusion_models.__file__)'
"$environment_root/bin/python" -m pytest -q
"$environment_root/bin/ruff" check .
"$environment_root/bin/ruff" format --check .
