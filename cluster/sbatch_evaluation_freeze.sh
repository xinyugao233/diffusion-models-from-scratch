#!/usr/bin/env bash
#SBATCH --job-name=spectral_eval_freeze
#SBATCH --partition=logical_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00

set -euo pipefail
job_scratch=${SLURM_TMPDIR:-/tmp/${USER}/spectral-eval-freeze-${SLURM_JOB_ID}}
export TMPDIR="$job_scratch/tmp"
export XDG_CACHE_HOME="$job_scratch/.cache"
export TORCH_HOME="$job_scratch/torch"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME"
repository=/home/xggh8/projects/diffusion-models-from-scratch
environment_root=/home/xggh8/data/diffusion-models-from-scratch/envs/torch-2.13.0
output=/home/xggh8/data/diffusion-models-from-scratch/spectral-boundary-study/evaluation-freeze/try01
: "${RUN_COMMIT:?RUN_COMMIT is required}"
[[ "$(git -C "$repository" rev-parse HEAD)" == "$RUN_COMMIT" ]]
[[ -z "$(git -C "$repository" status --porcelain)" ]]
[[ ! -e "$output" ]]
cd "$repository"
"$environment_root/bin/python" scripts/materialize_evaluation_freeze.py \
  --config configs/spectral_boundary_20k.json --output-dir "$output"
