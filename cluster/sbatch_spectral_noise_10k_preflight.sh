#!/usr/bin/env bash
#SBATCH --job-name=study2_10k_preflight
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -euo pipefail
repository=/home/xggh8/projects/diffusion-models-from-scratch-noise
environment_root=/home/xggh8/data/diffusion-models-from-scratch/envs/torch-2.13.0
final_root=/home/xggh8/data/diffusion-models-from-scratch/spectral-boundary-noise/study2_10k/preflight/${SLURM_JOB_ID}
job_scratch=${SLURM_TMPDIR:-/tmp/${USER}/study2-10k-preflight-${SLURM_JOB_ID}}
work_root="$job_scratch/output"
: "${RUN_COMMIT:?RUN_COMMIT is required}"
[[ "$(git -C "$repository" rev-parse HEAD)" == "$RUN_COMMIT" ]]
[[ -z "$(git -C "$repository" status --porcelain)" ]]
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
  "$HF_DATASETS_CACHE" "$PIP_CACHE_DIR" "$work_root" "$final_root"
stage_results() { rsync -a "$work_root/" "$final_root/"; }
trap stage_results EXIT
cd "$repository"
"$environment_root/bin/python" -m compileall -q src scripts
"$environment_root/bin/python" -m pytest -q
"$environment_root/bin/ruff" check .
"$environment_root/bin/ruff" format --check .
"$environment_root/bin/python" scripts/check_spectral_noise_10k_protocol.py \
  --config configs/spectral_boundary_noise_10k.json \
  > "$work_root/protocol_check.json"
for condition in baseline moving_narrow moving_broad; do
  "$environment_root/bin/python" scripts/train_spectral_noise_10k.py \
    --config configs/spectral_boundary_noise_10k.json \
    --condition "$condition" --seed 0 --segment-end 2 \
    --validation-sample --output-dir "$work_root/$condition"
done
touch "$work_root/COMPLETED"
