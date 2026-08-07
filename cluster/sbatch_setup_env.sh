#!/usr/bin/env bash
#SBATCH --job-name=ddpm_env
#SBATCH --partition=logical_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

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

if [[ -f "$environment_root/READY" ]]; then
  echo "Environment already ready: $environment_root"
  exit 0
fi
if [[ -e "$environment_root" ]]; then
  echo "Refusing to overwrite incomplete environment: $environment_root" >&2
  exit 1
fi

mkdir -p "$(dirname "$environment_root")"
/home/xggh8/miniconda3/bin/python3 -m venv "$environment_root"
"$environment_root/bin/python" -m pip install --upgrade pip
"$environment_root/bin/python" -m pip install \
  --requirement "$repository/requirements-hpc.txt"
"$environment_root/bin/python" -m pip install --no-deps --editable "$repository"
"$environment_root/bin/python" -m pip freeze > "$environment_root/pip-freeze.txt"
"$environment_root/bin/python" - <<'PY'
import torch
import torchvision

print(f"torch={torch.__version__}")
print(f"torchvision={torchvision.__version__}")
print(f"cuda_build={torch.version.cuda}")
PY
touch "$environment_root/READY"
