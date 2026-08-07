#!/usr/bin/env bash
#SBATCH --job-name=ddpm_env
#SBATCH --partition=logical_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

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
