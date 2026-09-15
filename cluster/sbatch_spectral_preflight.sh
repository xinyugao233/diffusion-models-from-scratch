#!/usr/bin/env bash
#SBATCH --job-name=spectral20k_preflight
#SBATCH --partition=logical_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail
job_scratch=${SLURM_TMPDIR:-/tmp/${USER}/spectral20k-preflight-${SLURM_JOB_ID}}
export TMPDIR="$job_scratch/tmp"
export XDG_CACHE_HOME="$job_scratch/.cache"
export TORCH_HOME="$job_scratch/torch"
repository=/home/xggh8/projects/diffusion-models-from-scratch
environment_root=/home/xggh8/data/diffusion-models-from-scratch/envs/torch-2.13.0
evaluation_deps=/home/xggh8/data/diffusion-models-from-scratch/eval-deps/torch-fidelity-0.4.0
: "${RUN_COMMIT:?RUN_COMMIT is required}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME"
[[ "$(git -C "$repository" rev-parse HEAD)" == "$RUN_COMMIT" ]]
[[ -z "$(git -C "$repository" status --porcelain)" ]]
[[ -f "$evaluation_deps/READY" ]]
cd "$repository"
PYTHONPATH="$evaluation_deps" "$environment_root/bin/python" \
  scripts/check_spectral_20k_protocol.py
