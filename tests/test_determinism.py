"""Reproducibility across fresh processes, hash seeds, and worker counts."""

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import polars as pl
import pytest
import torch

from determinism import configure_determinism

PROJECT = Path(__file__).resolve().parents[1]


def test_reseeding_replays_library_randomness():
    def draw():
        pixels = np.zeros((8, 8), dtype=np.float32)
        cv2.randu(pixels, np.array([0.0]), np.array([1.0]))
        return (
            random.random(),
            np.random.random(8).tolist(),
            pixels.tolist(),
            pl.Series(range(20)).sample(n=10, shuffle=True).to_list(),
            torch.rand(8).tolist(),
        )

    configure_determinism(123)
    first = draw()
    draw()  # Advancing RNG state must be undone by reseeding.
    configure_determinism(123)
    assert draw() == first
    configure_determinism(124)
    assert draw() != first
    assert cv2.getNumThreads() == 1
    assert not cv2.ocl.useOpenCL()
    assert not cv2.useOptimized()


def cached_dataset(root: Path, reverse: bool) -> None:
    """Build a complete offline cache whose file order differs between runs."""
    train_ids = [f"dish_train_{index}" for index in range(4)]
    test_ids = [f"dish_test_{index}" for index in range(8)]
    ids = train_ids + test_ids
    if reverse:
        ids.reverse()
        train_ids.reverse()
        test_ids.reverse()
    metadata = root / "metadata"
    metadata.mkdir(parents=True)
    (metadata / "dish_metadata_cafe1.csv").write_text(
        "".join(f"{dish},123.456789,250,10.123,20.456,30.789\n" for dish in ids)
    )
    (metadata / "dish_metadata_cafe2.csv").write_text("")
    splits = root / "dish_ids" / "splits"
    splits.mkdir(parents=True)
    for name, values in (("train", train_ids), ("test", test_ids)):
        (splits / f"rgb_{name}_ids.txt").write_text("\n".join(values))
    images = root / "images"
    images.mkdir()
    for index, dish in enumerate(sorted(test_ids)):
        frame = np.zeros((320, 320, 3), dtype=np.uint8)
        cv2.circle(frame, (160, 160), 130, (230, 230, 230), -1)
        cv2.circle(frame, (160, 160), 20 + index * 8, (20, 80, 140), -1)
        assert cv2.imwrite(str(images / f"{dish}.png"), frame)


def invoke(*args: str, hash_seed: str, workers: str) -> None:
    subprocess.run(
        [sys.executable, str(PROJECT / "eval.py"), *args],
        cwd=PROJECT,
        env={**os.environ, "PYTHONHASHSEED": hash_seed, "POLARS_MAX_THREADS": workers},
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize("backend", ["mean", "classical"])
def test_preparation_and_predictions_repeat_across_processes(tmp_path, backend):
    manifests, results = [], []
    for index, (hash_seed, workers) in enumerate((("1", "1"), ("999", "4"))):
        data = tmp_path / f"data-{index}"
        output = tmp_path / f"run-{index}"
        cached_dataset(data, reverse=bool(index))
        invoke(
            "prepare",
            "--data-dir",
            str(data),
            "--n",
            "4",
            "--seed",
            "19",
            hash_seed=hash_seed,
            workers=workers,
        )
        manifests.append((data / "manifest.json").read_bytes())
        invoke(
            "run",
            "--data-dir",
            str(data),
            "--predictor",
            backend,
            "--limit",
            "3",
            "--output",
            str(output),
            hash_seed=hash_seed,
            workers=workers,
        )
        config = json.loads((output / "config.json").read_text())
        manifest = json.loads(manifests[-1])
        assert config["sample_ids"] == [row["id"] for row in manifest["samples"][:3]]
        assert config["determinism"] == {
            "seed": 19,
            "opencv_threads": 0,
            "opencv_opencl": False,
            "opencv_optimized": False,
            "torch_deterministic_algorithms": True,
            "cublas_workspace_config": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG", ":4096:8"
            ),
        }
        rows = json.loads((output / "predictions.json").read_text())
        for row in rows:
            assert row["status"] == "ok"
            del row["latency_seconds"]
        summary = json.loads((output / "summary.json").read_text())
        del summary["mean_latency_seconds"]
        results.append((rows, summary))
    assert manifests[0] == manifests[1]
    assert results[0] == results[1]

    # A different preparation seed actually changes the selected evaluation set.
    other = tmp_path / "different-seed"
    cached_dataset(other, reverse=False)
    invoke(
        "prepare",
        "--data-dir",
        str(other),
        "--n",
        "4",
        "--seed",
        "20",
        hash_seed="1",
        workers="1",
    )
    changed = json.loads((other / "manifest.json").read_text())
    original = json.loads(manifests[0])
    assert [row["id"] for row in changed["samples"]] != [
        row["id"] for row in original["samples"]
    ]
