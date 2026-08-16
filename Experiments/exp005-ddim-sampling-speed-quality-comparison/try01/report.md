# EXP005 Try 01: DDIM Sampling Speed–Quality Comparison

## Status

`COMPLETED` — H100 job `16409278` finished with exit `0:0`; all frozen
provenance, call-count, shape, finiteness, repeatability, timing, and visual
gates passed.

## Goal and hypothesis tested

Compare DDPM-1000 with deterministic DDIM-100/50/25 using the same 50k EMA
checkpoint and the same materialized 16-image initial-noise batch. The frozen
hypothesis predicted exact reduced NFEs, deterministic repeatability, faster
synchronized H100 medians, and potentially visible degradation at fewer steps.

## Configuration and provenance

- Plan: `Experiments/exp005-ddim-sampling-speed-quality-comparison/plan.md`
- Config: `configs/ddim_comparison.json`
- Config SHA-256:
  `80c6e27776ff6aa839145fa09cae006a17dbbc1a3ae3d138bf6379fe387a1108`
- Run commit: `20c6ea9fd8b558016a1e995abb7db0ea723962fc`
- Checkpoint step/hash: `50000`,
  `f6b64f7028432ac1690029564610918e68e0c45b5188a97961863eadd79b17b9`
- Initial seeds: `1000..1015`
- DDPM reverse seed: `109`
- GPU: `NVIDIA H100 NVL` on node `g033`
- Environment: Python `3.13.11`, PyTorch `2.13.0+cu130`, CUDA `13.0`,
  cuDNN `92000`

The checkpoint and both configs were independently hashed after the run. The
Slurm stdout, staged log, and comparison JSON are byte-identical with SHA-256
`97aaec5bdb831d11b92ab1621aacdd9ee147b8da68aeb8fb6320cd1c057c4a62`.

## Execution

- Job: `16409278`
- State/exit: `COMPLETED`, `0:0`
- Start/end: `2026-08-12T11:43:08` / `2026-08-12T11:43:33`
- Scheduler elapsed: `25 s`
- Resources: one H100, eight CPUs, 64 GiB requested memory
- Warm-up: one untimed model call before all conditions
- Timing: CUDA synchronized immediately before and after each trajectory
- Repeats: three complete trajectories per condition

## Results

| Sampler | NFE | Mean runtime (s) | Median runtime (s) | Median speedup | Throughput (images/s) | Algorithm deterministic |
|---|---:|---:|---:|---:|---:|---|
| DDPM-1000 | 1000 | `4.611460` | `4.611370` | `1.0000x` | `3.4697` | no |
| DDIM-100 | 100 | `0.450763` | `0.450342` | `10.2397x` | `35.5286` | yes, `eta=0` |
| DDIM-50 | 50 | `0.224621` | `0.224743` | `20.5184x` | `71.1923` | yes, `eta=0` |
| DDIM-25 | 25 | `0.112438` | `0.112395` | `41.0284x` | `142.3556` | yes, `eta=0` |

Every repeat used the exact expected call count. All tensors have shape
`[16,3,32,32]`, contain only finite values, and repeat bitwise within each
condition. Exact timings, schedules, output hashes, system identity, and
artifact hashes are in `results/comparison.json` and `results/manifest.json`.

## Paired initial conditions and DDIM determinism

DDPM is intrinsically stochastic, but repeated runs were bitwise reproducible
under the frozen reverse RNG seed. DDIM with `eta=0` requires no reverse-step
random draws and is deterministic conditional on the initial `x_T`.

The experiment shared one materialized initial-noise batch across all sampler
conditions. The run persisted seed provenance but did not persist an explicit
hash of that batch; a hash was reconstructed locally from the frozen generation
function and seeds. The reconstructed batch SHA-256 is
`ea6c2359cdb092aa49c1429d74b185ca2a8168e016fce47cfd16ae30bd30768a`.
This derived hash is not represented as primary run evidence.

## Figures and visual inspection

- `figures/ddpm_ddim_headline.png`: all four paired 4x4 grids
- `figures/ddpm_1000_grid.png`
- `figures/ddim_100_grid.png`
- `figures/ddim_50_grid.png`
- `figures/ddim_25_grid.png`
- `figures/runtime_comparison_raw_job16409278.png`: immutable raw job figure
- `figures/runtime_comparison.png`: corrected figure regenerated from raw JSON

The grids contain recognizable CIFAR-like structure without catastrophic
artifacts. DDIM-100/50/25 preserve highly similar cellwise content, with no
clear monotonic 100-to-25 visual deterioration in this small fixed set. Their
distinct output hashes rule out exact duplication. DDPM differs because its
reverse trajectory consumes seeded stochastic noise. Pairing, ordering, color,
and display clamping appear correct.

The raw runtime plot clipped its longest annotation. A plotting-only code fix
and focused regression test corrected the derived presentation figure without
rerunning a sampler or changing any numerical result.

## Interpretation and limitations

The exact checkpoint-specific NFE and H100 speed claims pass. The images
support only a conservative statement that recognizable structure remains for
these 16 seeds. The predicted monotonic qualitative degradation is not clearly
observed. No FID/KID, distribution-level quality, held-out behavior,
generalization, likelihood, cross-hardware timing, or universal DDIM claim is
supported.

The missing run-recorded initial tensor hash is a provenance limitation,
although same-object use is explicit in the exact run code. Future comparison
scripts should store both the batch hash and per-seed tensor hashes.

## Exact evidence paths

- Local compact evidence:
  `Experiments/exp005-ddim-sampling-speed-quality-comparison/try01/`
- Remote immutable try:
  `/home/xggh8/data/diffusion-models-from-scratch/exp005-ddim-comparison/try01`
- Slurm stdout:
  `/home/xggh8/data/diffusion-models-from-scratch/slurm-logs/exp005-ddim-comparison-16409278.out`
- Parent checkpoint:
  `/home/xggh8/data/diffusion-models-from-scratch/exp004-full-cifar10-ddpm-training/try01/full/checkpoints/checkpoint_step_050000.pt`

## Validation

- `.venv/bin/python -m pytest -q`: `72 passed`
- `.venv/bin/ruff check .`: passed
- `.venv/bin/ruff format --check .`: passed
- `git diff --check`: passed
- Repository Markdown/local-link validation: passed
- Corrected runtime-figure regression test: passed

## Next step

Stop technical model development. Proceed to recruiter-facing presentation,
CV bullets, and interview review without EDM, flow matching, additional
training, FID/KID, or a 100k extension.
