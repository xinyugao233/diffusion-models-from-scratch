# EXP005 DDIM sampling speed-quality comparison

## Status

`READY` — deterministic DDIM implementation and local validation pass;
checkpoint execution is pending.

## Purpose

Complete the final technical milestone by comparing the trained 50k EMA model
under ancestral DDPM and deterministic reduced-step DDIM sampling.

## Main question

How much synchronized H100 sampling time is saved at 100, 50, and 25 DDIM
steps, and what visible structure remains for the same 16 initial tensors?

## Design

- Checkpoint: immutable `EXP004/try01` step 50k EMA.
- Settings: DDPM-1000; DDIM-100, DDIM-50, DDIM-25.
- Seeds: initial `1000..1015`; DDPM reverse seed `109`.
- Repeats: three synchronized timings per setting.
- Quality: fixed-seed visual comparison only; no FID/KID.
- Execution: one guarded H100 Slurm job.

## Try index

| Try | Status | Conclusion |
|---|---|---|
| `try01` | `READY` | Implementation validated; execution pending |

## Evidence

- Plan: `Experiments/exp005-ddim-sampling-speed-quality-comparison/plan.md`
- Hypothesis: `Experiments/exp005-ddim-sampling-speed-quality-comparison/hypothesis.md`
- Try: `Experiments/exp005-ddim-sampling-speed-quality-comparison/try01/report.md`
