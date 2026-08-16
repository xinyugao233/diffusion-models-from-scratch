# Current Context

Milestones 1 and 2 are complete at baseline commit
`26e735e597001251d92075d5bf02fb5649851800`. The private GitHub repository and
remote CPU CI are operational; the Milestone 2 workflow passed 22 tests plus
Ruff lint and formatting. Its Node.js deprecation warning is recorded as
deferred maintenance debt.

Milestone 3 and `EXP001/try01` are complete. On one local CPU thread, the fixed
synthetic gate achieved late/early mean-loss ratio `0.010670` over 40 steps and
the frozen 16-image CIFAR-10 gate achieved `0.199254` over 500 steps; both beat
the preregistered maximum ratio `0.80`, produced only finite records, and
completed normally. The fixed subset, hashes, raw logs, loss CSV, visually
inspected curve, and exact environment are durable under the try folder and
summarized in `Reports/milestone_03.md`.

Milestone 4 is published at commit `2f13b3f` and GitHub Actions run
`31152749677` passed 41 tests plus lint and formatting. Explicit EMA and a
versioned structured checkpoint preserve model, EMA, AdamW, global step,
configuration, and Python/torch RNG. The corrected deterministic validation
found exact equality between 100 uninterrupted steps and reconstructed 50+50
training for every state tensor, the full loss trajectory, and the next random
draw/loss. The RNG-omission control differed as required. Try01's mapping-
container comparator failure is preserved rather than overwritten.

Milestone 5 and `EXP003/try01` are complete locally. The exact posterior,
epsilon-to-clean conversion, deterministic final step, supplied reverse noise,
seeded complete loop, and EMA-model interface pass 13 focused tests. One
random-weight CPU smoke run completed exactly 1,000 calls in 12.6002 seconds
with a finite four-image result and six visually inspected trajectory states.

The valid Milestone 5 conclusion is limited to structural DDPM sampling
correctness.
The noise-like figure is expected from random weights and provides no evidence
about generation quality. No training, trained-checkpoint evaluation, FID,
DDIM, GPU, or dataset work was performed. It was published as commit `7061038`,
and GitHub Actions run `31154154963` passed 54 tests plus lint and formatting.

Milestone 6 and `EXP004/try01` are complete. Commit `7ad4524` passed GitHub
Actions run `31155794394`; Hellbender jobs `15914422`, `15914482`, `15938195`,
and `15943245` passed Gate A, the new-process Gate B resume, and the fresh 50k
run. The full trajectory contains exactly 50,000 contiguous finite records.
The last-1,000/first-1,000 mean-loss ratio is `0.465576`, passing the frozen
`0.9` gate, and the visually inspected fixed-seed 10k/25k/50k progression
reaches recognizable CIFAR-like animal and vehicle forms. The exact result and
two preserved infrastructure failures are documented in
`Reports/milestone_06.md`. Stop at 50k; no extension or DDIM implementation is
authorized without a new reviewed plan.

Milestone 7 and `EXP005/try01` are complete. Deterministic eta-zero DDIM, the
frozen DDPM-1000 versus DDIM-100/50/25 protocol, and guarded execution were run
from commit `20c6ea9fd8b558016a1e995abb7db0ea723962fc`. H100 job `16409278`
completed with exit `0:0`; exact NFEs, finite output shapes, repeated hashes,
checkpoint/config identities, and staged artifacts validate. Synchronized
median runtimes were `4.611370`, `0.450342`, `0.224743`, and `0.112395` seconds,
for DDIM speedups of `10.2397x`, `20.5184x`, and `41.0284x`. Visual inspection
found recognizable CIFAR-like structure in every grid and no clear monotonic
DDIM-100-to-25 degradation for the frozen 16 seeds. The technical DDPM/DDIM MVP
is complete; next work is recruiter-facing presentation, not another model or
training extension.
