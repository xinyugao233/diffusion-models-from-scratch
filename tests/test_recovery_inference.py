"""Maintenance checks use synthetic fixtures, never historical run evidence."""

import copy
from dataclasses import asdict

import pytest
import torch
from torch import nn

from diffusion_models.checkpointing import (
    CheckpointError,
    load_training_checkpoint,
    save_training_checkpoint,
    states_exactly_equal,
)
from diffusion_models.ema import ExponentialMovingAverage
from diffusion_models.full_training import AppendOnlyJsonlLogger
from diffusion_models.inference import NORMALIZATION, load_bundle, sample_bundle
from diffusion_models.models import CIFAR10UNet, smoke_unet_config
from diffusion_models.recovery import (
    atomic_publish,
    recover_run,
    sha256,
    stage_checkpoint,
)


def components():
    torch.manual_seed(123)
    model = nn.Linear(2, 2)
    return (
        model,
        torch.optim.AdamW(model.parameters(), lr=0.01),
        ExponentialMovingAverage(model, 0.9),
    )


def update(model, optimizer, ema):
    optimizer.zero_grad()
    loss = model(torch.randn(2, 2)).square().mean()
    loss.backward()
    optimizer.step()
    ema.update(model)
    return loss.item()


def checkpoint_fixture(path, model, optimizer, ema):
    return save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        ema=ema,
        global_step=ema.num_updates,
        configuration={"fixture": True},
    )


def test_recovered_continuation_exact_and_originals_preserved(tmp_path):
    model, optimizer, ema = components()
    log = tmp_path / "metrics.jsonl"
    logger = AppendOnlyJsonlLogger(log, expected_last_step=0)
    losses = []
    for step in range(1, 5):
        losses.append(update(model, optimizer, ema))
        logger.log(step, loss=losses[-1])
        if step == 2:
            checkpoint = checkpoint_fixture(
                tmp_path / "checkpoint.pt", model, optimizer, ema
            )
    expected = copy.deepcopy(
        [model.state_dict(), optimizer.state_dict(), ema.state_dict()]
    )
    next_random = torch.randn(5)
    with log.open("ab") as handle:
        handle.write(b'{"step":5,broken')
    hashes = sha256(log), sha256(checkpoint)
    target = tmp_path / "recovered"
    manifest = recover_run(
        checkpoint, log, target, expected_configuration={"fixture": True}
    )
    assert manifest["malformed_uncommitted_tail"]
    assert (sha256(log), sha256(checkpoint)) == hashes
    restored_model, restored_optimizer, restored_ema = components()
    loaded = load_training_checkpoint(
        target / "checkpoints/checkpoint.pt",
        model=restored_model,
        optimizer=restored_optimizer,
        ema=restored_ema,
        expected_configuration={"fixture": True},
    )
    logger = AppendOnlyJsonlLogger(
        target / "metrics.jsonl", expected_last_step=loaded.global_step
    )
    for step in range(3, 5):
        loss = update(restored_model, restored_optimizer, restored_ema)
        assert loss == losses[step - 1]
        logger.log(step, loss=loss)
    actual = [
        restored_model.state_dict(),
        restored_optimizer.state_dict(),
        restored_ema.state_dict(),
    ]
    assert states_exactly_equal(expected, actual)
    assert torch.equal(torch.randn(5), next_random)
    assert (sha256(log), sha256(checkpoint)) == hashes


@pytest.mark.parametrize(
    "failure", ["prefix", "checkpoint", "configuration", "destination"]
)
def test_recovery_refuses_bad_inputs(tmp_path, failure):
    model, optimizer, ema = components()
    update(model, optimizer, ema)
    checkpoint = checkpoint_fixture(tmp_path / "state.pt", model, optimizer, ema)
    log = tmp_path / "log"
    log.write_text('{"step":1,"loss":1.0}\n')
    destination = tmp_path / "new"
    config = {"fixture": True}
    if failure == "prefix":
        log.write_text('{"step":2,"loss":1.0}\n')
    elif failure == "checkpoint":
        checkpoint.write_bytes(b"broken synthetic checkpoint")
    elif failure == "configuration":
        config = {"fixture": False}
    else:
        destination.mkdir()
        (destination / "existing").write_text("preserve")
    with pytest.raises((ValueError, CheckpointError, FileExistsError)):
        recover_run(checkpoint, log, destination, expected_configuration=config)


def test_atomic_failure_and_staging_do_not_publish_partial_files(tmp_path):
    destination = tmp_path / "snapshot.pt"

    def interrupted(handle):
        handle.write(b"partial")
        raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError):
        atomic_publish(destination, interrupted, lambda path: None)
    assert not destination.exists()
    model, optimizer, ema = components()
    source = checkpoint_fixture(tmp_path / "source.pt", model, optimizer, ema)
    stage_checkpoint(source, destination)
    assert sha256(source) == sha256(destination)
    with pytest.raises(FileExistsError):
        stage_checkpoint(source, destination)
    assert not list(tmp_path.glob("*.partial"))


def test_synthetic_bundle_repeatability_schema_and_missing_weights(tmp_path):
    config = smoke_unet_config()
    model = CIFAR10UNet(config)
    payload = {
        "schema_version": 1,
        "normalization": NORMALIZATION,
        "architecture": asdict(config),
        "ema_state_dict": model.state_dict(),
        "diffusion": {
            "schedule": "linear",
            "num_steps": 4,
            "beta_start": 0.0001,
            "beta_end": 0.02,
        },
        "training_step": 0,
        "source_sha256": "synthetic fixture",
        "attribution": "UNTRAINED TEST FIXTURE",
    }
    path = tmp_path / "synthetic.pt"
    torch.save(payload, path)
    a, _ = sample_bundle(path, steps=2, count=1, seed=7)
    b, _ = sample_bundle(path, steps=2, count=1, seed=7)
    assert torch.equal(a, b)
    assert not load_bundle(path)[0].training
    with pytest.raises(FileNotFoundError, match="missing"):
        load_bundle(tmp_path / "missing.pt")
    payload["architecture"]["base_channels"] = 64
    torch.save(payload, tmp_path / "incompatible.pt")
    with pytest.raises(ValueError, match="Incompatible"):
        load_bundle(tmp_path / "incompatible.pt")


def test_step_zero_recovery_allows_strict_logger_start(tmp_path):
    model, optimizer, ema = components()
    checkpoint = checkpoint_fixture(tmp_path / "zero.pt", model, optimizer, ema)
    log = tmp_path / "interrupted.jsonl"
    log.write_bytes(b'{"step":1,broken')
    target = tmp_path / "continued"
    manifest = recover_run(
        checkpoint, log, target, expected_configuration={"fixture": True}
    )
    assert manifest["checkpoint_step"] == 0
    logger = AppendOnlyJsonlLogger(target / "metrics.jsonl", expected_last_step=0)
    logger.log(1, loss=0.5)
    assert logger.last_step == 1
