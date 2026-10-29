"""Relay finished Babel trial histories and status snapshots to W&B locally."""

import argparse
import hashlib
import shlex
import subprocess
import time
from pathlib import Path

from publish_sweep import publish
from sync_babble import upload_run


def watch(plan, results, jobs, identity):
    host = "cyrils@login.babel.cs.cmu.edu"
    remote = "/home/cyrils/macs/optuna-pilot/"
    ssh = ["ssh", "-i", str(identity), "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
    previous = None
    queue_drained = False
    deadline = time.monotonic() + 3 * 60 * 60
    while time.monotonic() < deadline:
        subprocess.run(
            [
                "rsync",
                "-az",
                "--exclude=*.pt",
                "--exclude=study.db*",
                "--exclude=wandb/",
                "-e",
                shlex.join(ssh),
                f"{host}:{remote}",
                str(results.resolve()) + "/",
            ],
            check=True,
        )
        for backbone in ("resnet", "swin", "dino"):
            for summary in sorted((results / backbone).glob("trial-*/summary.json")):
                if (summary.parent / "wandb.json").exists():
                    upload_run(summary.parent)
        digest = hashlib.sha256()
        for pattern in ("*/trial-*/config.json", "*/trial-*/summary.json"):
            for path in sorted(results.glob(pattern)):
                digest.update(str(path).encode())
                digest.update(path.read_bytes())
        current = digest.hexdigest()
        if current != previous:
            publish(plan, results)
            previous = current
        if queue_drained:
            print(
                "All submitted jobs left the queue; final fetched records uploaded.",
                flush=True,
            )
            return
        queue = subprocess.run(
            [*ssh, host, "squeue --noheader --jobs=" + ",".join(jobs) + " --format=%i"],
            check=True,
            capture_output=True,
            text=True,
        )
        if not queue.stdout.strip():
            # Fetch again after observing termination, avoiding a last-summary race.
            queue_drained = True
            continue
        print("Remaining jobs: " + queue.stdout.replace("\n", " "), flush=True)
        time.sleep(60)
    raise TimeoutError(
        "Relay reached its three-hour limit; inspect Slurm and fetched records"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("jobs", nargs="+")
    parser.add_argument(
        "--identity", type=Path, default=Path.home() / ".ssh/id_ed25519"
    )
    args = parser.parse_args()
    watch(args.plan, args.results, args.jobs, args.identity)
