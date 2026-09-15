#!/usr/bin/env bash
#SBATCH --job-name=spectral20k_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-11

set -euo pipefail
conditions=(baseline moving_narrow moving_broad static_matched)
condition=${conditions[$((SLURM_ARRAY_TASK_ID / 3))]}
pair=$((SLURM_ARRAY_TASK_ID % 3))
repository=/home/xggh8/projects/diffusion-models-from-scratch
environment_root=/home/xggh8/data/diffusion-models-from-scratch/envs/torch-2.13.0
evaluation_deps=/home/xggh8/data/diffusion-models-from-scratch/eval-deps/torch-fidelity-0.4.0
run_root=/home/xggh8/data/diffusion-models-from-scratch/spectral-boundary-study-20k/try01/${condition}/pair-${pair}
final_root="$run_root/evaluation"
job_scratch=${SLURM_TMPDIR:-/tmp/${USER}/spectral20k-eval-${SLURM_JOB_ID}}
work_output="$job_scratch/output"
: "${RUN_COMMIT:?RUN_COMMIT is required}"
[[ "$(git -C "$repository" rev-parse HEAD)" == "$RUN_COMMIT" ]]
[[ -z "$(git -C "$repository" status --porcelain)" ]]
[[ -f "$run_root/COMPLETED" ]]
[[ -f "$evaluation_deps/READY" ]]
[[ ! -e "$final_root/COMPLETED" ]]
export TMPDIR="$job_scratch/tmp"
export XDG_CACHE_HOME="$job_scratch/.cache"
export TORCH_HOME="$job_scratch/torch"
export PYTHONPATH="$evaluation_deps"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$final_root"
stage_results() { [[ ! -d "$work_output" ]] || rsync -a "$work_output/" "$final_root/"; }
trap stage_results EXIT
cd "$repository"
"$environment_root/bin/python" scripts/evaluate_spectral_20k.py \
  --config configs/spectral_boundary_20k.json --condition "$condition" --pair "$pair" \
  --training-dir "$run_root" --output-dir "$work_output"
touch "$work_output/COMPLETED"
