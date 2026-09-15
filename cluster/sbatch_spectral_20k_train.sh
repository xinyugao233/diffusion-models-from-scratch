#!/usr/bin/env bash
#SBATCH --job-name=spectral20k_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-11

set -euo pipefail
conditions=(baseline moving_narrow moving_broad static_matched)
condition=${conditions[$((SLURM_ARRAY_TASK_ID / 3))]}
pair=$((SLURM_ARRAY_TASK_ID % 3))
repository=/home/xggh8/projects/diffusion-models-from-scratch
environment_root=/home/xggh8/data/diffusion-models-from-scratch/envs/torch-2.13.0
final_root=/home/xggh8/data/diffusion-models-from-scratch/spectral-boundary-study-20k/try01/${condition}/pair-${pair}
job_scratch=${SLURM_TMPDIR:-/tmp/${USER}/spectral20k-${SLURM_JOB_ID}}
work_output="$job_scratch/output"
: "${RUN_COMMIT:?RUN_COMMIT is required}"
[[ "$(git -C "$repository" rev-parse HEAD)" == "$RUN_COMMIT" ]]
[[ -z "$(git -C "$repository" status --porcelain)" ]]
[[ ! -e "$final_root/COMPLETED" ]]
export TMPDIR="$job_scratch/tmp"
export XDG_CACHE_HOME="$job_scratch/.cache"
export TORCH_HOME="$job_scratch/torch"
export MPLCONFIGDIR="$job_scratch/matplotlib"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" "$work_output" "$final_root"
stage_results() { rsync -a "$work_output/" "$final_root/"; }
trap stage_results EXIT
cd "$repository"
"$environment_root/bin/python" scripts/train_full_cifar10.py \
  --config configs/spectral_boundary_20k.json --stage stage_20k \
  --condition "$condition" --pair "$pair" --segment-end 20000 \
  --output-dir "$work_output"
touch "$work_output/COMPLETED"
