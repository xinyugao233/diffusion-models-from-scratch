#!/usr/bin/env bash
#SBATCH --job-name=ddpm_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00

set -euo pipefail

job_scratch=${SLURM_TMPDIR:-/tmp/${USER}/ddpm-${SLURM_JOB_ID}}
export TMPDIR="$job_scratch/tmp"
export XDG_CACHE_HOME="$job_scratch/.cache"
export TORCH_HOME="$job_scratch/torch_cache"
export MPLCONFIGDIR="$job_scratch/matplotlib"
export WANDB_DIR="$job_scratch/wandb"
export HF_HOME="$job_scratch/hf"
export TRANSFORMERS_CACHE="$job_scratch/hf/transformers"
export HF_DATASETS_CACHE="$job_scratch/hf/datasets"
export PIP_CACHE_DIR="$job_scratch/pip-cache"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

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

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: sbatch_train.sh <gate_a|gate_b|full> <segment_end> [resume_checkpoint_name]" >&2
  exit 2
fi
stage=$1
segment_end=$2
resume_name=${3:-}
case "$stage" in
  gate_a|gate_b|full) ;;
  *) echo "Invalid stage: $stage" >&2; exit 2 ;;
esac
if [[ ! "$segment_end" =~ ^[0-9]+$ ]] || [[ "$segment_end" -le 0 ]]; then
  echo "segment_end must be a positive integer" >&2
  exit 2
fi

repository=/home/xggh8/projects/diffusion-models-from-scratch
environment_root=/home/xggh8/data/diffusion-models-from-scratch/envs/torch-2.13.0
persistent_root=/home/xggh8/data/diffusion-models-from-scratch/exp004-full-cifar10-ddpm-training/try01
persistent_stage="$persistent_root/$stage"
run_directory="$job_scratch/run"

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
if [[ -f "$persistent_stage/COMPLETED_SEGMENT_${segment_end}" ]]; then
  echo "Refusing to rerun completed segment $stage/$segment_end" >&2
  exit 1
fi

mkdir -p "$run_directory"
if [[ -d "$persistent_stage" ]]; then
  rsync -a "$persistent_stage/" "$run_directory/"
fi
mkdir -p "$run_directory/logs"

stage_outputs() {
  mkdir -p "$persistent_stage"
  rsync -a "$run_directory/" "$persistent_stage/"
}
trap stage_outputs EXIT

resume_arguments=()
if [[ -n "$resume_name" ]]; then
  resume_path="$run_directory/checkpoints/$resume_name"
  if [[ ! -f "$resume_path" ]]; then
    echo "Missing staged resume checkpoint: $resume_path" >&2
    exit 1
  fi
  resume_arguments=(--resume "$resume_path")
fi

cd "$repository"
"$environment_root/bin/python" scripts/train_full_cifar10.py \
  --config configs/full_cifar10_training.json \
  --stage "$stage" \
  --output-dir "$run_directory" \
  --segment-end "$segment_end" \
  "${resume_arguments[@]}" \
  2>&1 | tee -a "$run_directory/logs/train_segment_${segment_end}.log"

touch "$run_directory/COMPLETED_SEGMENT_${segment_end}"
