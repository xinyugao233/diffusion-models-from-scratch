# Portable inference and staged model card

**Availability:** the original trusted 50K training checkpoint was recovered
from the authorized Hellbender path, matched to its recorded checksum, and
exported locally. Publication and a redistribution/license decision remain
pending. There is no public download URL. No new license is inferred.

## Source model

- Project: Diffusion models from scratch; historical EXP004/try01.
- Data: full CIFAR-10 training split; 32×32 RGB, model-space `[-1,1]`.
- Architecture: custom primary U-Net, 12,852,547 parameters.
- Training step: 50,000; inference uses EMA weights.
- Schedule: linear DDPM, 1,000 steps, beta `1e-4..0.02`, float32.
- Source checkpoint SHA-256:
  `f6b64f7028432ac1690029564610918e68e0c45b5188a97961863eadd79b17b9`.
- Source identity: `EXP004/try01/full/checkpoints/checkpoint_step_050000.pt`.
- Historical benchmark: [EXP005 manifest](../../Experiments/exp005-ddim-sampling-speed-quality-comparison/try01/results/manifest.json).
- Intended use: research/educational inference and implementation review.
  No FID/KID parity, privacy, deployment, or cross-hardware performance claim.

## Export and inference

Only use a trusted source identified by its independently recorded checksum.
The exporter uses restricted `weights_only=True` loading and requires the
primary architecture and recorded float32/linear schedule contract. It does
not execute arbitrary pickle globals or silently replace missing weights.

```bash
python scripts/export_inference.py --checkpoint /path/to/checkpoint_step_050000.pt --expected-sha256 f6b64f7028432ac1690029564610918e68e0c45b5188a97961863eadd79b17b9 --output /path/to/new-ema-50000-v1.pt --attribution 'Diffusion models from scratch; EXP004/try01; rights pending owner confirmation'
python scripts/sample.py --checkpoint /path/to/new-ema-50000-v1.pt --output-dir outputs/samples-01 --device cpu --sampler ddim --inference-steps 50 --sample-count 16 --seed 0
```

Schema v1 stores EMA tensors, exact architecture, diffusion schedule, training
step, source checksum/metadata, normalization and attribution. Optimizer/RNG
training state is excluded. `--sampler ddpm` uses all schedule steps;
`--inference-steps` controls DDIM only. Output is `samples.pt`, `samples.png`,
and a manifest recording checkpoint, seed, environment, NFE and scope.
The output directory must not exist.

The portable CLI uses evaluation mode, explicit RNG and deterministic
algorithms. Repeatability is tested with a clearly labeled untrained synthetic
bundle on CPU; bitwise agreement across environments/devices is not promised.
A CUDA environment may also require `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
Real weights were exported and structurally loaded; **new trained-model
sampling and GPU timing were not run**. Historical samples remain historical.

## Release staging (not uploaded)

The local handoff directory `../external-evidence/diffusion` (relative to the
maintenance checkout) holds the original checkpoint, exported bundle,
`SHA256SUMS`, model card and historical sample tensors. Exact current hashes
and sizes are in [artifacts.json](artifacts.json).
A release must include the model card and checksums and resolve model/code
redistribution rights before uploading. The export checksum identifies the
produced file, not a portable guarantee of identical serialization bytes.
