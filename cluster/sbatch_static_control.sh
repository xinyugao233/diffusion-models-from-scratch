#!/usr/bin/env bash
#SBATCH --job-name=static_spectral_control
#SBATCH --partition=logical_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00

set -euo pipefail

job_scratch=${SLURM_TMPDIR:-/tmp/${USER}/static-control-${SLURM_JOB_ID}}
export TMPDIR="$job_scratch/tmp"
export XDG_CACHE_HOME="$job_scratch/.cache"
export TORCH_HOME="$job_scratch/torch_cache"
export MPLCONFIGDIR="$job_scratch/matplotlib"
export WANDB_DIR="$job_scratch/wandb"
export HF_HOME="$job_scratch/hf"
export TRANSFORMERS_CACHE="$job_scratch/hf/transformers"
export HF_DATASETS_CACHE="$job_scratch/hf/datasets"
export PIP_CACHE_DIR="$job_scratch/pip-cache"

mkdir -p \
  "$TMPDIR" \
  "$XDG_CACHE_HOME" \
  "$TORCH_HOME" \
  "$MPLCONFIGDIR" \
  "$WANDB_DIR" \
  "$HF_HOME" \
  "$TRANSFORMERS_CACHE" \
  "$HF_DATASETS_CACHE" \
  "$PIP_CACHE_DIR"

repository=/home/xggh8/projects/diffusion-models-from-scratch
environment_root=/home/xggh8/data/diffusion-models-from-scratch/envs/torch-2.13.0
artifact_root=/home/xggh8/data/diffusion-models-from-scratch/static-control
attempt=${ATTEMPT:-job-${SLURM_JOB_ID}}
job_output="$job_scratch/output"
final_output="$artifact_root/$attempt"

if [[ ! -f "$environment_root/READY" ]]; then
  echo "Missing ready environment: $environment_root" >&2
  exit 1
fi
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
if [[ -e "$final_output" ]]; then
  echo "Refusing to overwrite existing attempt: $final_output" >&2
  exit 1
fi

mkdir -p "$job_output"
cd "$repository"
"$environment_root/bin/python" scripts/materialize_static_spectral_control.py \
  --radial-power configs/cifar10_50k_radial_power.csv \
  --output-dir "$job_output"

mkdir -p "$artifact_root"
mv "$job_output" "$final_output"
printf '%s\n' "$RUN_COMMIT" > "$final_output/run_commit.txt"
printf '%s\n' "$final_output"
