# EXP005 DDIM sampling speed-quality comparison

## Status

`COMPLETED` — H100 job `16409278` passed the frozen comparison and artifact
validation. DDIM-100/50/25 used exact NFEs and were `10.2397x`, `20.5184x`, and
`41.0284x` faster than DDPM-1000 by synchronized median runtime.

## Research question

How much synchronized H100 sampling time is saved at 100, 50, and 25 DDIM
steps, and what visible structure remains for the same 16 initial tensors?

## Motivation and larger program

Milestone 6 established a trained CIFAR-10 DDPM checkpoint. EXP005 completes
the technical MVP by applying deterministic reduced-step DDIM to that exact
EMA denoiser without retraining.

## Design and invariants

- Baseline: stochastic DDPM-1000 with reverse seed `109`.
- Intervention: deterministic DDIM-100/50/25 with `eta=0`.
- Checkpoint: immutable `EXP004/try01` step-50k EMA.
- Noise control: one materialized batch from seeds `1000..1015` reused by all
  settings.
- Repeats: three synchronized complete trajectories per setting.
- Hardware: one H100 NVL.
- Quality evidence: all 16 fixed rows; no FID/KID or cherry-picking.

## Try index

| Try | Status | Conclusion |
|---|---|---|
| `try01` | `COMPLETED` | Exact NFEs, deterministic repeats, faster DDIM medians, and recognizable fixed-seed samples passed. |

## Current conclusion

| Sampler | NFE | Median runtime | Speedup vs DDPM | Throughput |
|---|---:|---:|---:|---:|
| DDPM-1000 | 1000 | `4.611370 s` | `1.0000x` | `3.4697 images/s` |
| DDIM-100 | 100 | `0.450342 s` | `10.2397x` | `35.5286 images/s` |
| DDIM-50 | 50 | `0.224743 s` | `20.5184x` | `71.1923 images/s` |
| DDIM-25 | 25 | `0.112395 s` | `41.0284x` | `142.3556 images/s` |

All four outputs are finite `[16,3,32,32]` tensors and repeat bitwise within
condition. Visual inspection found recognizable CIFAR-like structure in all
settings and no clear monotonic DDIM-100-to-25 degradation in this small fixed
set. The result is checkpoint-, seed-, batch-, and H100-specific.

DDPM is intrinsically stochastic, but repeated runs were bitwise reproducible
under the frozen reverse RNG seed. DDIM with `eta=0` requires no reverse-step
random draws and is deterministic conditional on the initial `x_T`.

## Immediate next action

Stop technical model development and prepare recruiter-facing presentation,
CV bullets, and interview review.

## Evidence

- [Plan](plan.md)
- [Preregistered hypothesis](hypothesis.md)
- [Try report](try01/report.md)
- [Milestone report](../../Reports/milestone_07.md)
- [Machine-readable results](try01/results/comparison.json)
- [Paired comparison figure](try01/figures/ddpm_ddim_headline.png)
- [Runtime figure](try01/figures/runtime_comparison.png)
