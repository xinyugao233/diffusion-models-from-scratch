# Understanding EMA and Reproducible Training Checkpoints

## What EMA is

An exponential moving average keeps a smoothed copy of each training
parameter. After optimizer update `s`, this project applies

```math
\theta_{ema,s}=\gamma\theta_{ema,s-1}+(1-\gamma)\theta_s.
```

The initial EMA state is an exact detached clone of the model. A decay near one
changes slowly; a smaller decay follows the training model more closely.

## Why EMA is not optimized directly

The loss and backward pass produce gradients for the training model. AdamW
uses those gradients and its momentum statistics to update that model. EMA is a
deterministic summary of the resulting parameter trajectory, not another loss-
optimized network. Passing EMA tensors to the optimizer would mix two different
update rules and destroy the intended smoothing.

EMA tensors are therefore detached, have `requires_grad=False`, and update only
after `optimizer.step()`.

## Why EMA helps diffusion evaluation

Stochastic gradient updates make individual training parameters fluctuate.
EMA suppresses short-term oscillations and often gives a more stable denoiser
for evaluation and sampling. It does not guarantee better samples, and this
milestone does not sample; it only prepares the state that a future sampler can
use through `ema.copy_to(model)` or `ema.model_state_dict()`.

## Parameter and buffer policy

Floating trainable parameters receive the EMA equation. Buffers are not
optimizer parameters: examples include BatchNorm running statistics and integer
counters. This implementation copies every named buffer exactly from the
training model after each optimizer step. It does not average integer counters
or silently ignore evaluation-relevant state.

EMA update and restoration require exact parameter/buffer names, shapes,
dtypes, and devices. This prevents a checkpoint from silently attaching state
to a different architecture.

## Why model weights alone cannot resume training

The next update depends on more than weights:

- AdamW's moving gradient and squared-gradient estimates;
- the global step and any step-dependent future schedules;
- EMA weights and update count;
- random streams that choose timesteps, noise, augmentation, or sampling order;
- the exact configuration and implementation identity.

Loading only model weights starts a related but different optimization
trajectory.

## What optimizer state contains conceptually

AdamW stores per-parameter first-moment estimates, second-moment estimates, and
step counters, plus parameter-group hyperparameters. These statistics determine
the scale and direction of the next adaptive update. `optimizer.state_dict()`
serializes them without serializing the optimizer Python object itself.

## Why RNG state matters

DDPM training draws a timestep and Gaussian noise for every image. If a resumed
run begins from a different torch RNG state, its very next input/target pair is
different even when model and optimizer weights match. Object reconstruction
also consumes random numbers during parameter initialization, so RNG must be
restored after objects are rebuilt and checkpoint state is loaded.

This checkpoint always records Python and torch CPU RNG state. It can record
named `torch.Generator` states. It records CUDA RNG only for explicitly used
CUDA devices and NumPy RNG only when NumPy randomness is explicitly enabled.
Milestone 4 uses neither CUDA nor NumPy randomness.

## Reproducible sampling versus reproducible training

Reproducible sampling means the same trained model and sampling seed generate
the same trajectory. Reproducible training means every future minibatch,
gradient, optimizer update, EMA update, and state agrees after interruption.
Training reproducibility requires substantially more state.

## Inference export versus training checkpoint

An inference export may contain only model or EMA weights plus enough
architecture configuration to reconstruct the network. A resumable training
checkpoint additionally contains optimizer, step, RNG, full training
configuration, and provenance. This project uses a structured checkpoint for
resume and exposes EMA model state separately for future inference.

## What `state_dict` means

A PyTorch `state_dict` is a mapping from stable names or optimizer identifiers
to tensors and simple metadata. Saving state dictionaries instead of entire
objects makes reconstruction explicit: code creates the expected class and
then validates and loads its state. This is easier to inspect and less coupled
to Python object pickling than serializing a whole model instance.

## Checkpoint save/load sequence

Save after a completed optimizer and EMA update:

```text
loss.backward
optimizer.step
ema.update(model)
increment global step
capture RNG
save structured state dictionaries and metadata
```

Resume in this order:

```text
reconstruct model, optimizer, and EMA
load model state
load EMA state
load optimizer state
restore global step and configuration
restore RNG after reconstruction
continue with the next training step
```

The checkpoint writer refuses to overwrite an existing path, and the loader
rejects missing keys, unsupported versions, incompatible tensors, and
configuration mismatches.

## Common resume bugs

- Saving model weights but forgetting AdamW state.
- Saving EMA weights but forgetting its update count or buffer state.
- Updating EMA before rather than after the optimizer update.
- Accidentally including EMA tensors in the optimizer.
- Restoring RNG before reconstructing a randomly initialized model.
- Restoring global RNG while forgetting a dedicated `torch.Generator`.
- Capturing one CUDA device while training consumes another device's RNG.
- Loading a checkpoint under a different schedule or architecture silently.
- Averaging integer buffers as though they were floating parameters.
- Saving entire Python model objects instead of explicit state dictionaries.
- Overwriting the only good checkpoint with a partial write.
- Comparing only final loss instead of every state and the next random draw.
- Claiming cross-hardware reproducibility from a same-machine CPU test.

## Interview questions and concise answers

1. **What is EMA?** A recursively weighted average of past training-model
   parameters, biased toward recent values.
2. **Who updates EMA?** Explicit post-optimizer code; neither autograd nor the
   optimizer updates it.
3. **Why initialize EMA from the model?** It gives a defined state at step zero
   without a startup bias toward zeros.
4. **Why update after `optimizer.step()`?** The EMA for step `s` should include
   the newly produced step-`s` model parameters.
5. **Why can EMA improve diffusion samples?** It filters noisy parameter
   oscillations and often provides a more stable denoiser for evaluation.
6. **What is inside AdamW state?** Per-parameter moment estimates and counters,
   plus parameter-group hyperparameters.
7. **Why save the global step?** Future schedules, logging, checkpoint cadence,
   and EMA policies may depend on it.
8. **Why restore RNG after reconstruction?** Model construction consumes random
   draws that must not displace the saved training stream.
9. **Why compare the next noise draw?** Matching stored weights alone does not
   prove that the next stochastic training example will match.
10. **What makes a checkpoint structured?** It stores named state dictionaries,
    scalars, configuration, RNG, and metadata rather than a whole Python object.
11. **Should buffers be exponentially averaged?** Not automatically. This
    implementation copies them exactly, which is defined for floating and
    integer buffers.
12. **Does exact CPU resume imply exact GPU resume?** No. Device kernels,
    versions, and distributed execution can introduce different determinism
    constraints.
13. **What is the difference between a checkpoint and an EMA export?** A
    checkpoint resumes optimization; an EMA export is sufficient only for
    evaluation once architecture information is available.
