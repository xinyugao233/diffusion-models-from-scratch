#!/usr/bin/env bash
#SBATCH --job-name=spectral_eval_setup
#SBATCH --partition=logical_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00

set -euo pipefail
job_scratch=${SLURM_TMPDIR:-/tmp/${USER}/spectral-eval-setup-${SLURM_JOB_ID}}
export TMPDIR="$job_scratch/tmp"
export XDG_CACHE_HOME="$job_scratch/.cache"
export PIP_CACHE_DIR="$job_scratch/pip-cache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR"
repository=/home/xggh8/projects/diffusion-models-from-scratch
training_env=/home/xggh8/data/diffusion-models-from-scratch/envs/torch-2.13.0
dependency_root=/home/xggh8/data/diffusion-models-from-scratch/eval-deps/torch-fidelity-0.4.0
asset_root=/home/xggh8/data/diffusion-models-from-scratch/eval-assets/torch-fidelity-0.4.0
: "${RUN_COMMIT:?RUN_COMMIT is required}"
[[ "$(git -C "$repository" rev-parse HEAD)" == "$RUN_COMMIT" ]]
[[ -z "$(git -C "$repository" status --porcelain)" ]]
[[ ! -e "$dependency_root" ]]
[[ ! -e "$asset_root" ]]

packages="$job_scratch/packages"
assets="$job_scratch/assets"
mkdir -p "$packages" "$assets"
"$training_env/bin/python" -m pip install --no-deps --target "$packages" \
  -r "$repository/requirements-evaluation.txt"
weights="$assets/weights-inception-2015-12-05-6726825d.pth"
curl -fL --retry 3 \
  https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/weights-inception-2015-12-05-6726825d.pth \
  -o "$weights"
PYTHONPATH="$packages" "$training_env/bin/python" - <<'PY'
import scipy
import torch_fidelity
import tqdm
print("torch_fidelity", torch_fidelity.__version__)
print("scipy", scipy.__version__)
print("tqdm", tqdm.__version__)
PY
mkdir -p "$(dirname "$dependency_root")" "$(dirname "$asset_root")"
mv "$packages" "$dependency_root"
mv "$assets" "$asset_root"
sha256sum "$asset_root/weights-inception-2015-12-05-6726825d.pth"
find "$dependency_root" -type f -print0 | sort -z | xargs -0 sha256sum > "$asset_root/dependency_files.sha256"
printf '%s\n' "$RUN_COMMIT" > "$asset_root/setup_commit.txt"
touch "$dependency_root/READY" "$asset_root/READY"
