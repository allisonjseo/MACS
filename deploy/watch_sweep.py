"""Relay live Babel epoch metrics and status snapshots to W&B locally."""

import argparse
import hashlib
import shlex
import subprocess
import time
from pathlib import Path

from live_sweep import LiveRelay
from publish_sweep import publish

SYNC_FILTERS = [
    "--exclude=wandb/",
    "--include=*/",
    "--include=*.json",
    "--include=*.jsonl",
    "--include=*.csv",
    "--include=*.out",
    "--include=*.err",
    "--exclude=*",
]


def watch(plan, results, jobs, identity):
    relay = LiveRelay()
    try:
        watch_loop(plan, results, jobs, identity, relay)
    finally:
        relay.close()


def watch_loop(plan, results, jobs, identity, relay):
    host = "cyrils@login.babel.cs.cmu.edu"
    remote = "/home/cyrils/macs/optuna-pilot/"
    ssh = ["ssh", "-i", str(identity), "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
    previous = None
    queue_drained = False
    deadline = time.monotonic() + 3 * 60 * 60
    while time.monotonic() < deadline:
        fetched = subprocess.run(
            [
                "rsync",
                "-az",
                *SYNC_FILTERS,
                "-e",
                shlex.join(ssh),
                f"{host}:{remote}",
                str(results.resolve()) + "/",
            ],
        )
        if fetched.returncode:
            print(
                f"Fetch failed ({fetched.returncode}); retrying in 15 seconds without closing online runs.",
                flush=True,
            )
            time.sleep(15)
            continue
        for backbone in ("resnet", "swin", "dino"):
            for output in sorted((results / backbone).glob("trial-*")):
                relay.update(output)
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
            relay.close()
            print(
                "All submitted jobs left the queue; final fetched records uploaded.",
                flush=True,
            )
            return
        queue = subprocess.run(
            [*ssh, host, "squeue --noheader --jobs=" + ",".join(jobs) + " --format=%i"],
            capture_output=True,
            text=True,
        )
        if queue.returncode:
            print(
                f"Queue check failed: {queue.stderr.strip()}; retrying in 15 seconds.",
                flush=True,
            )
            time.sleep(15)
            continue
        if not queue.stdout.strip():
            # Fetch again after observing termination, avoiding a last-summary race.
            queue_drained = True
            continue
        print("Remaining jobs: " + queue.stdout.replace("\n", " "), flush=True)
        time.sleep(15)
    relay.close()
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
