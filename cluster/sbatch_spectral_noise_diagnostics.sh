#!/usr/bin/env bash
#SBATCH --job-name=spectral_noise_diag
#SBATCH --partition=logical_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail
job_scratch=${SLURM_TMPDIR:-/tmp/${USER}/spectral-noise-diag-${SLURM_JOB_ID}}
export TMPDIR="$job_scratch/tmp"
export XDG_CACHE_HOME="$job_scratch/.cache"
export TORCH_HOME="$job_scratch/torch"
export MPLCONFIGDIR="$job_scratch/matplotlib"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$MPLCONFIGDIR"
repository=/home/xggh8/projects/diffusion-models-from-scratch-noise
environment_root=/home/xggh8/data/diffusion-models-from-scratch/envs/torch-2.13.0
output=/home/xggh8/data/diffusion-models-from-scratch/spectral-boundary-noise/diagnostics/try01
: "${RUN_COMMIT:?RUN_COMMIT is required}"
[[ "$(git -C "$repository" rev-parse HEAD)" == "$RUN_COMMIT" ]]
[[ -z "$(git -C "$repository" status --porcelain)" ]]
[[ ! -e "$output" ]]
cd "$repository"
"$environment_root/bin/python" scripts/diagnose_spectral_noise_schedule.py \
  --config configs/spectral_boundary_noise_smoke.json --output-dir "$output"
