"""Atomic epoch checkpoints and reproducible RNG/history restoration."""

import json
import os
import random

import numpy as np
import torch


def rng_state(generator):
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader": generator.get_state(),
    }


def restore_rng(state, generator):
    random.setstate(state["python"])
    name, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state(
        (name, np.asarray(keys, dtype=np.uint32), pos, has_gauss, cached)
    )
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("CUDA checkpoint requires CUDA; no device fallback")
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader"])


def atomic_checkpoint(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def publish_best(output, name):
    source = output / name
    if source.parent.resolve() != output.resolve() or not source.is_file():
        raise ValueError(
            "Committed best checkpoint is missing or outside the run directory"
        )
    temporary = output / "best.pt.link.tmp"
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(name)
    temporary.replace(output / "best.pt")
    # Keep the old best until last.pt has committed its replacement.
    for old in output.glob(".best-epoch-*.pt"):
        if old.name != name:
            old.unlink()


def read_history(path):
    if not path.exists():
        return []
    rows = [
        json.loads(line)
        for line in path.read_bytes().splitlines(keepends=True)
        if line.endswith(b"\n")
    ]
    if [r["epoch"] for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("Epoch history must be contiguous and unique")
    return rows


def recover_history(path, saved):
    """Checkpoint is the commit point; repair an interrupted JSON append."""
    rows = read_history(path)
    if len(rows) > len(saved):
        raise ValueError("Epoch log is ahead of the committed checkpoint")
    for actual, expected in zip(rows, saved, strict=False):
        for key in ("epoch", "train", "validation", "learning_rates"):
            if actual[key] != expected[key]:
                raise ValueError("Epoch log disagrees with committed checkpoint")
    if len(rows) < len(saved):
        rows.extend(saved[len(rows) :])
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows)
    )
    temporary.replace(path)
    return rows
