"""Run all three encoders serially on the same allocated GPU and time each one."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    manifest, output_root, *overrides = sys.argv[1:]
    project = Path(__file__).resolve().parents[1]
    workspace = project.parent
    environment = os.environ.copy()
    for name, directory in {
        "HF_HOME": workspace / "cache/huggingface",
        "TORCH_HOME": workspace / "cache/torch",
        "TORCHINDUCTOR_CACHE_DIR": workspace / "cache/inductor",
        "WANDB_CACHE_DIR": workspace / "cache/wandb",
    }.items():
        environment.setdefault(name, str(directory))
    job_id = os.environ["SLURM_JOB_ID"]
    results = []
    for model in ("resnet", "swin", "dino"):
        output = Path(output_root) / f"{job_id}-{model}"
        command = [
            sys.executable,
            "train.py",
            "fit",
            "--config",
            f"configs/{model}.toml",
            "--manifest",
            manifest,
            "--output",
            str(output),
            *overrides,
        ]
        print(f"Starting {model}: {command}", flush=True)
        started = time.perf_counter()
        result = subprocess.run(command, cwd=project, env=environment)
        row = {
            "model": model,
            "wall_seconds": time.perf_counter() - started,
            "returncode": result.returncode,
            "output": str(output),
        }
        results.append(row)
        Path(output_root, f"{job_id}-suite-summary.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )
        print(json.dumps(row), flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
