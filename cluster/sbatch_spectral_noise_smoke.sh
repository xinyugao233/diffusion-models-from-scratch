#!/usr/bin/env bash
#SBATCH --job-name=spectral_noise_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-2

set -euo pipefail
conditions=(noise_baseline noise_moving_narrow noise_moving_broad)
condition=${conditions[$SLURM_ARRAY_TASK_ID]}
repository=/home/xggh8/projects/diffusion-models-from-scratch-noise
environment_root=/home/xggh8/data/diffusion-models-from-scratch/envs/torch-2.13.0
final_root=/home/xggh8/data/diffusion-models-from-scratch/spectral-boundary-noise/smoke/try01/${condition}
job_scratch=${SLURM_TMPDIR:-/tmp/${USER}/spectral-noise-smoke-${SLURM_JOB_ID}}
work_output="$job_scratch/output"
: "${RUN_COMMIT:?RUN_COMMIT is required}"
[[ "$(git -C "$repository" rev-parse HEAD)" == "$RUN_COMMIT" ]]
[[ -z "$(git -C "$repository" status --porcelain)" ]]
[[ ! -e "$final_root/COMPLETED" ]]
export TMPDIR="$job_scratch/tmp"
export XDG_CACHE_HOME="$job_scratch/.cache"
export TORCH_HOME="$job_scratch/torch"
export MPLCONFIGDIR="$job_scratch/matplotlib"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" "$final_root"
stage_results() { [[ ! -d "$work_output" ]] || rsync -a "$work_output/" "$final_root/"; }
trap stage_results EXIT
cd "$repository"
"$environment_root/bin/python" scripts/train_spectral_noise_smoke.py \
  --config configs/spectral_boundary_noise_smoke.json \
  --condition "$condition" --output-dir "$work_output"
touch "$work_output/COMPLETED"
