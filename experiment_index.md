# Experiment Index

| Experiment ID | Title | Status | Best Try | Main Finding | Link |
|---|---|---|---|---|---|
| EXP001 | Epsilon prediction learnability | COMPLETED | try01 | Synthetic and fixed-16 CIFAR-10 epsilon loss passed frozen reduction gates. | `Experiments/exp001-epsilon-prediction-learnability/try01/report.md` |
| EXP002 | Deterministic checkpoint resume | COMPLETED | try02 | Corrected try02 matched model, EMA, AdamW, losses, step, and next RNG draw exactly. | `Experiments/exp002-deterministic-checkpoint-resume/try02/report.md` |
| EXP003 | DDPM reverse sampling | COMPLETED | try01 | Posterior gates and a finite seeded 1,000-call random-model reverse chain passed. | `Experiments/exp003-ddpm-reverse-sampling/try01/report.md` |
| EXP004 | Full CIFAR-10 DDPM training | COMPLETED | try01 | 50k H100 run passed stability/loss gates and produced recognizable fixed-seed EMA samples. | `Experiments/exp004-full-cifar-10-ddpm-training/try01/report.md` |
| EXP005 | DDIM sampling speed-quality comparison | COMPLETED | try01 | Exact NFEs and repeated hashes passed; DDIM-100/50/25 measured 10.2397x/20.5184x/41.0284x median H100 speedups with recognizable fixed-seed samples. | `Experiments/exp005-ddim-sampling-speed-quality-comparison/try01/report.md` |
