#!/usr/bin/env bash
#SBATCH --job-name=spectral_noise_smoke_generic
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail
conditions=(noise_baseline noise_moving_narrow noise_moving_broad)
repository=/home/xggh8/projects/diffusion-models-from-scratch-noise
environment_root=/home/xggh8/data/diffusion-models-from-scratch/envs/torch-2.13.0
final_root=/home/xggh8/data/diffusion-models-from-scratch/spectral-boundary-noise/smoke/try01
job_scratch=${SLURM_TMPDIR:-/tmp/${USER}/spectral-noise-smoke-generic-${SLURM_JOB_ID}}
work_root="$job_scratch/output"
: "${RUN_COMMIT:?RUN_COMMIT is required}"
[[ "$(git -C "$repository" rev-parse HEAD)" == "$RUN_COMMIT" ]]
[[ -z "$(git -C "$repository" status --porcelain)" ]]
[[ ! -e "$final_root/COMPLETED" ]]
export TMPDIR="$job_scratch/tmp"
export XDG_CACHE_HOME="$job_scratch/.cache"
export TORCH_HOME="$job_scratch/torch"
export MPLCONFIGDIR="$job_scratch/matplotlib"
export WANDB_DIR="$job_scratch/wandb"
export HF_HOME="$job_scratch/hf"
export TRANSFORMERS_CACHE="$job_scratch/hf/transformers"
export HF_DATASETS_CACHE="$job_scratch/hf/datasets"
export PIP_CACHE_DIR="$job_scratch/pip-cache"
mkdir -p \
  "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" \
  "$WANDB_DIR" "$HF_HOME" "$TRANSFORMERS_CACHE" \
  "$HF_DATASETS_CACHE" "$PIP_CACHE_DIR" "$final_root"
stage_results() { [[ ! -d "$work_root" ]] || rsync -a "$work_root/" "$final_root/"; }
trap stage_results EXIT
cd "$repository"
for condition in "${conditions[@]}"; do
  "$environment_root/bin/python" scripts/train_spectral_noise_smoke.py \
    --config configs/spectral_boundary_noise_smoke.json \
    --condition "$condition" --output-dir "$work_root/$condition"
done
touch "$work_root/COMPLETED"
