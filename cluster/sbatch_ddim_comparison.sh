#!/usr/bin/env bash
#SBATCH --job-name=ddim_compare
#SBATCH --partition=gpu
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00

set -euo pipefail

job_scratch=${SLURM_TMPDIR:-/tmp/${USER}/ddim-compare-${SLURM_JOB_ID}}
export TMPDIR="$job_scratch/tmp"
export XDG_CACHE_HOME="$job_scratch/.cache"
export TORCH_HOME="$job_scratch/torch_cache"
export MPLCONFIGDIR="$job_scratch/matplotlib"
export HF_HOME="$job_scratch/hf"
export PIP_CACHE_DIR="$job_scratch/pip-cache"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

repository=/home/xggh8/projects/diffusion-models-from-scratch
environment_root=/home/xggh8/data/diffusion-models-from-scratch/envs/torch-2.13.0
checkpoint_source=/home/xggh8/data/diffusion-models-from-scratch/exp004-full-cifar10-ddpm-training/try01/full/checkpoints/checkpoint_step_050000.pt
persistent_try=/home/xggh8/data/diffusion-models-from-scratch/exp005-ddim-comparison/try01
run_directory="$job_scratch/run"
checkpoint_local="$job_scratch/checkpoint_step_050000.pt"

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
if [[ -e "$persistent_try" ]]; then
  echo "Refusing to overwrite existing EXP005 try01: $persistent_try" >&2
  exit 1
fi
if [[ ! -f "$checkpoint_source" ]]; then
  echo "Missing frozen checkpoint: $checkpoint_source" >&2
  exit 1
fi

mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" \
  "$HF_HOME" "$PIP_CACHE_DIR" "$run_directory/logs"
cp "$checkpoint_source" "$checkpoint_local"

stage_outputs() {
  mkdir -p "$persistent_try"
  rsync -a "$run_directory/" "$persistent_try/"
}
trap stage_outputs EXIT

cd "$repository"
"$environment_root/bin/python" scripts/run_ddim_comparison.py \
  --config configs/ddim_comparison.json \
  --checkpoint "$checkpoint_local" \
  --output-dir "$run_directory" \
  2>&1 | tee "$run_directory/logs/ddim_comparison.log"

touch "$run_directory/COMPLETED"
