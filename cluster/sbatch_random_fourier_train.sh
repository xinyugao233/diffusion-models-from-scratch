#!/usr/bin/env bash
#SBATCH --job-name=fourier_pair
#SBATCH --partition=gpu
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-2

set -euo pipefail

job_scratch=${SLURM_TMPDIR:-/tmp/${USER}/fourier-pair-${SLURM_JOB_ID}}
export TMPDIR="$job_scratch/tmp"
export XDG_CACHE_HOME="$job_scratch/.cache"
export TORCH_HOME="$job_scratch/torch_cache"
export MPLCONFIGDIR="$job_scratch/matplotlib"
export WANDB_DIR="$job_scratch/wandb"
export HF_HOME="$job_scratch/hf"
export TRANSFORMERS_CACHE="$job_scratch/hf/transformers"
export HF_DATASETS_CACHE="$job_scratch/hf/datasets"
export PIP_CACHE_DIR="$job_scratch/pip-cache"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" \
  "$WANDB_DIR" "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" \
  "$PIP_CACHE_DIR"

repository=/home/xggh8/projects/diffusion-models-from-scratch
environment_root=/home/xggh8/data/diffusion-models-from-scratch/envs/torch-2.13.0
persistent_root=/home/xggh8/data/diffusion-models-from-scratch/random-cifar1k-spatial-fourier/try01
seed=${SLURM_ARRAY_TASK_ID}
persistent_seed="$persistent_root/seed_${seed}"
persistent_attempt="$persistent_seed/attempt_${SLURM_JOB_ID}"
run_directory="$job_scratch/run"
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
if find "$persistent_seed" -mindepth 2 -maxdepth 2 -name COMPLETED -print -quit \
  2>/dev/null | grep -q .; then
  echo "Refusing to overwrite completed paired seed $seed" >&2
  exit 1
fi
mkdir -p "$run_directory"
stage_outputs() {
  mkdir -p "$persistent_attempt"
  rsync -a "$run_directory/" "$persistent_attempt/"
}
trap stage_outputs EXIT

cd "$repository"
"$environment_root/bin/python" scripts/train_random_cifar1k_fourier_pair.py \
  --config configs/random_cifar1k_spatial_fourier.json \
  --seed "$seed" \
  --output-dir "$run_directory" \
  2>&1 | tee "$run_directory/train.log"
