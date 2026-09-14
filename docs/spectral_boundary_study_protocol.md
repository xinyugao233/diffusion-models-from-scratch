# Moving Spectral Boundary Study Protocol

This protocol was frozen before intervention training. The implementation has
passed its baseline-equivalence checks, but no learned-model evidence yet
supports the intervention.

The machine-readable source of truth is
`configs/spectral_boundary_study.json`. The primary treatments use
`tau=ln(2)`, `tau=ln(4)`, and `weight_floor=0.1` over the complete DDPM
schedule. Wild-middle-only weighting is reserved for a later ablation.

## Pre-training gates

1. Recompute the exact E006 radial statistic on all 50K CIFAR-10 training
   images through Slurm.
2. Apply the preregistered spectrum-selection rule in the config. Preserve the
   E006 1K result as provenance regardless of the choice.
3. Materialize and test the static weight vector matched to the narrow moving
   treatment.
4. Pin the held-out denoising examples, timesteps, and noise, plus the exact
   nearest-neighbor evaluator identity.
5. Verify that evaluation and logging use isolated random-number generators.
6. Run the full test/lint preflight at the exact clean training commit.

Passing these gates authorizes neither the 12 long training runs nor any change
to the preregistered analysis. Training begins only as an explicit subsequent
step.

## Interpretation

H1 is evaluated using paired quality and ordinary held-out denoising
trajectories. H2 compares the moving narrow treatment with its static matched
control. H3 is evaluated only at matched generative quality; quality changes
must not be used as a proxy for memorization.
