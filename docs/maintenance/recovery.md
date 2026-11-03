# Interrupted-run recovery

This maintenance changes persistence and explicit recovery only. Training loss,
noise, dataset, seeds, architecture, schedule, and budgets are unchanged.
Original logs/checkpoints are never truncated or edited. A training restart
still requires its own exact job/source/budget authorization.

For a trusted full-training checkpoint at step k, the offline CPU utility
checks configuration equality, config-file checksum in metadata, checkpoint
schema and tensors, and model/EMA/optimizer restoration. It requires complete,
contiguous finite-loss records 1..k. Later records and malformed trailing bytes
are retained separately as `interrupted_tail.bin`; they are not resumed.
The manifest records original source paths/hashes, metadata, configuration,
step, prefix hash, tail hash/size and whether the tail was malformed.

```bash
python scripts/recover_training_run.py --checkpoint /path/to/checkpoint_step_005000.pt --log /path/to/metrics.jsonl --config configs/full_cifar10_training.json --stage full --output-dir /path/to/new-continuation/full
```

The destination must be new or empty. Existing nonempty destinations, missing
prefix records, corrupt checkpoints and incompatible configurations are refused.
The strict production logger resumes only at the recovered checkpoint boundary.
The checkpoint itself is copied byte-for-byte to `checkpoints/`.
For step zero, no empty metrics file is created, so the strict logger can start
at zero. The original log must still exist and its entire tail is preserved.

The maintained launcher now requires `PERSISTENT_ROOT` explicitly. Point it at
a new run root or the parent of a recovered stage directory. It passes
`--durable-dir "${DURABLE_ROOT:-$persistent_stage/durable}"` to training.
Each checkpoint boundary stages a validated checkpoint and a separately
immutable log prefix, without relying on the EXIT trap. If the later log copy
fails, a complete checkpoint may exist without its log prefix; recover only
when the matching original log is also available.

Local persistence uses a sibling `.partial` file, flush/fsync, validation, then
atomic exclusive hard-link publication and directory fsync. Unsupported
filesystems fail rather than falling back to non-atomic overwriting. Partial
files are not named as completed snapshots. Existing snapshots are refused.
This retains the no-overwrite guarantee even if another writer wins the race.

Tests simulate a write failure and verify complete-file staging, then compare
uninterrupted and recovered toy continuations: parameters, EMA, optimizer,
loss sequence and the next RNG draw are exactly equal. Original hashes remain
unchanged. These establish **local state restoration and logical recovery**.
Node loss, SIGKILL, network-filesystem fsync semantics, quotas, and cluster
staging behavior have **not** been tested. No Slurm job was submitted.
