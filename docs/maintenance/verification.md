# Initial local maintenance verification — 2026-11-02

This record predates publication. See the [final review](publication-review.md)
and [maintenance PR](https://github.com/xinyugao233/diffusion-models-from-scratch/pull/1) for subsequent publication and hosted checks.

Local implementation is complete; publication, hosted CI and cluster durability
are separate unverified actions. This is maintenance evidence, not a new GPU
experiment. No historical numerical result or completed try was replaced.

## Baseline

Read-only `git ls-remote ... refs/heads/main` confirmed that GitHub main still
matched the starting commit below at handoff; no newer public work was replaced.

- Source checkout: `<original-checkout>`.
- Original branch `main`, clean; starting commit
  `ee09627f6d2a3edf9a7137b4765c1bbfd64e62a4`.
- Dedicated local clone, branch `codex/public-maintenance`; original checkout
  and the unrelated workspace research branch were preserved.
- Existing environment: Python 3.13.2, Torch 2.13.0, torchvision 0.28.0.
- Baseline command: `PYTHONPATH=src <original-checkout>/.venv/bin/python -m pytest -q`:
  **72 passed in 2.37 s**. The demo output collision and DDIM capture-index
  mismatch were confirmed in executable code.
- Declared compatibility at baseline: Python >=3.10, Torch >=2.2,
  torchvision >=0.17. This maintenance verifies the specific review stack
  below; it does not certify every allowed historical combination.

## Completed checklist

| Item | Status | Result |
|---|---|---|
| D1 | DONE | New 39-update public config plus one attention update; explicit new output directory; protected image outputs; literal demo command in CI |
| D2 | DONE | Mathematical state labels, independent analytic capture value, invalid-capture errors; final output/calls/RNG preserved |
| D3 | DONE locally | New-directory prefix recovery; original/tail hashes retained; atomic exclusive snapshots and checkpoint-boundary durable staging |
| D4 | DONE locally | EMA-only schema/export and portable CLI; actual 50K checkpoint verified and exported; release candidate staged |
| D5 | DONE | Float64 embedding conversion; float32/64 loop contract; temporary eval mode/restoration and dropout tests |
| R1 | DONE | CPU quickstart, explicit inference availability, larger derived figure labels, historical notes archived byte-for-byte |
| R3 | DONE locally | Source map, links, environment constraints and review metadata; remote changes not performed |
| External publication | BLOCKED | Requires action-specific authorization and owner license/redistribution decision |
| Cluster durability | NOT RUN | Local failure/staging fixtures do not establish node-loss or network-filesystem durability |

## Final review environment and commands

macOS 26.6.2 arm64; Python 3.13.2; Torch 2.15.0, torchvision 0.29.2,
NumPy 2.5.4. A **fresh isolated environment** installed only declared package
extras/dependencies with the review constraints. `pip check` reported no broken
requirements. [Install log](fresh-install.log).

```bash
python -m pip install -c constraints-review.txt -e '.[dev,report]'
python -m pytest -q
ruff check .
ruff format --check .
bash -n cluster/sbatch_train.sh
python scripts/check_document_links.py
python scripts/run_synthetic_optimization.py --config configs/public_cpu_demo.json --output-dir "${DEMO_OUTPUT:-outputs/cpu-demo}"
```

CPU tests: **83 passed**, including recovery/staging and inference fixtures.
[Pytest output](pytest-final.log) · [Ruff output](ruff-final.log).
The exact demo command passed: **39 epsilon updates + 1 diagnostic update**,
39 metric records, optimizer-loop runtime **1.8595 s**, late/early loss ratio
**0.0123566**. [Demo output](cpu-demo.log) · [repeat refusal](demo-repeat-refusal.log). This is a synthetic check, not
CIFAR-10 training or reproduction of EXP001. Reusing the same output directory
was checked to fail before optimization and preserve earlier output bytes.

All original EXP004/EXP005 files and original configs were compared against the
starting commit and remain byte-identical. The original runtime figure's hash
matches its archived raw-job image; derived presentation files are separately
identified. [Presentation hashes](presentation.json).

Recovered evidence: original 50K checkpoint, complete 50K-step log and all four
benchmark sample tensors verified; EMA-only export loaded structurally.
[Artifact inventory](artifacts.md). **No new trained-model inference, GPU
benchmark, FID evaluation, Slurm submission, push or upload was performed.**

## Review metadata staged, not applied

Description: “From-scratch PyTorch DDPM/DDIM with a custom CIFAR-10 U-Net,
EMA checkpoints, deterministic recovery tests and controlled sampling evidence.”
Topics: `pytorch`, `diffusion-models`, `ddpm`, `ddim`, `cifar10`, `reproducibility`.
The existing workflow now includes the installed README demo; hosted execution
for this local branch is pending. Local commits are the review artifact.

Remaining risks: historical GPU behavior was not rerun; CUDA/reduced precision
and node-loss persistence are not newly certified. License/redistribution and
public hosting remain owner decisions. See [recovery](recovery.md) and
[inference](inference.md) for the exact operational boundaries.

## Publication preparation

Original local commits and unredacted audit files are retained outside the
publication branch. Public log copies redact machine-specific paths; hashes,
commands and measured results are retained. New maintenance documents use
portable artifact identities. Historical manifests and completed experiment
records retain their original paths to preserve provenance.

[Independent final publication checks](publication-review.md).
