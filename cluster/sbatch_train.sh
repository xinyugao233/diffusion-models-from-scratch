#!/usr/bin/env bash
#SBATCH --job-name=ddpm_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00

set -euo pipefail

export TMPDIR="$SLURM_TMPDIR/tmp"
export XDG_CACHE_HOME="$SLURM_TMPDIR/.cache"
export TORCH_HOME="$SLURM_TMPDIR/torch_cache"
export MPLCONFIGDIR="$SLURM_TMPDIR/matplotlib"
export WANDB_DIR="$SLURM_TMPDIR/wandb"
export HF_HOME="$SLURM_TMPDIR/hf"
export TRANSFORMERS_CACHE="$SLURM_TMPDIR/hf/transformers"
export HF_DATASETS_CACHE="$SLURM_TMPDIR/hf/datasets"
export PIP_CACHE_DIR="$SLURM_TMPDIR/pip-cache"
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
run_directory="$SLURM_TMPDIR/run"

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
