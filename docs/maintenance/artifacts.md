# Artifact availability, 2026-11-02

[Exact local hashes, sizes and source paths](artifacts.json) distinguish original
bytes from new maintenance exports. All items below are available locally in
`../external-evidence/diffusion` relative to this checkout, and **not published**.

| Artifact | Verification and role |
|---|---|
| `checkpoint_step_050000.pt` | Matches historical `f6b64f70…b17b9`; restored restricted checkpoint schema, step 50,000, architecture and EMA validated during export |
| `metrics.jsonl` | 18,019,253 bytes; SHA-256 `1fbd7e08897f21e0c2fe056bef465e2d11e2a42d106907f04920ec2f0de4896f`; current remote/local bytes agree; contiguous steps 1..50,000; final loss 0.03368700295686722 |
| Four `ddpm_1000` / `ddim_100` / `ddim_50` / `ddim_25` sample tensors | All file hashes match EXP005's original manifest; run commit `20c6ea9fd8b558016a1e995abb7db0ea723962fc`, job 16409278 |
| `ema-50000-v1.pt` | New EMA-only export; source checksum and training identity embedded; exact output hash in JSON and staged `SHA256SUMS` |

For trusted local copies, `shasum -a 256 <file>` verifies exact bytes. Use
`python scripts/export_inference.py --help` and the [export command](inference.md)
to build a new bundle. Sampling requires the verified bundle; no substitute
untrained model is supplied as a trained release.

The historical runtime figure repair was already present at baseline:
`runtime_comparison_raw_job16409278.png` preserves the original manifest hash,
while `runtime_comparison.png` is the earlier derived rendering. Neither was
changed. New presentation images live under `figures/presentation_v2/` with
[presentation provenance](presentation.json), separate from original evidence.

The minimal release candidate contains only the EMA bundle, model card and
checksums. Original training/checkpoint/benchmark evidence remains in local
staging. Public hosting, code/model license selection, and redistribution
approval are unresolved. No guessed URL or public availability claim is made.
