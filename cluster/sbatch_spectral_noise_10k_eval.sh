#!/usr/bin/env bash
#SBATCH --job-name=study2_10k_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-8

set -euo pipefail
conditions=(baseline moving_narrow moving_broad)
condition=${conditions[$((SLURM_ARRAY_TASK_ID / 3))]}
seed=$((SLURM_ARRAY_TASK_ID % 3))
repository=/home/xggh8/projects/diffusion-models-from-scratch-noise
environment_root=/home/xggh8/data/diffusion-models-from-scratch/envs/torch-2.13.0
evaluation_deps=/home/xggh8/data/diffusion-models-from-scratch/eval-deps/torch-fidelity-0.4.0
run_root=/home/xggh8/data/diffusion-models-from-scratch/spectral-boundary-noise/study2_10k/try01/${condition}/seed${seed}
final_root="$run_root/evaluation"
job_scratch=${SLURM_TMPDIR:-/tmp/${USER}/study2-10k-eval-${SLURM_JOB_ID}}
work_root="$job_scratch/output"
: "${RUN_COMMIT:?RUN_COMMIT is required}"
[[ "$(git -C "$repository" rev-parse HEAD)" == "$RUN_COMMIT" ]]
[[ -z "$(git -C "$repository" status --porcelain)" ]]
[[ -f "$run_root/COMPLETED" ]]
[[ -f "$evaluation_deps/READY" ]]
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
export PYTHONPATH="$evaluation_deps"
mkdir -p \
  "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" \
  "$WANDB_DIR" "$HF_HOME" "$TRANSFORMERS_CACHE" \
  "$HF_DATASETS_CACHE" "$PIP_CACHE_DIR" "$work_root" "$final_root"
stage_results() { rsync -a "$work_root/" "$final_root/"; }
trap stage_results EXIT
cd "$repository"
"$environment_root/bin/python" scripts/evaluate_spectral_noise_10k.py \
  --config configs/spectral_boundary_noise_10k.json \
  --condition "$condition" --seed "$seed" \
  --training-dir "$run_root" --output-dir "$work_root"
touch "$work_root/COMPLETED"
