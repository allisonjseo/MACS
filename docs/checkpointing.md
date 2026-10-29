# Training checkpoints and preemption recovery

The trainer saves `last.pt` before the first epoch and after every completed
epoch. These checkpoints contain model buffers/weights, AdamW state, cosine
scheduler state, AMP scaler state, Python/NumPy/Torch CPU/CUDA RNG states, the
DataLoader generator, best-checkpoint bookkeeping, and committed epoch history.
They preserve the original maximum epochs and schedule horizon. There is no
weights-only warm start, optimizer reset, or new trial proposal on recovery.

Each checkpoint is written to a temporary file, flushed/fsynced, then atomically
replaces `last.pt`. A kill while writing leaves the previous checkpoint intact.
The selected model is stored in a separate immutable `.best-epoch-*.pt` file;
`best.pt` is an atomically updated symlink. The old best is retained until the
new `last.pt` commits its replacement. Recovery repairs the best pointer before
training, and removes unreferenced old best files. Inference still loads `best.pt`
using the original checkpoint format.

Recovery starts at the **next epoch after the last committed checkpoint**. An
unfinished epoch is repeated. Worker processes are recreated; deterministic
preprocessing and the restored DataLoader generator reproduce their seeds and
the next shuffle. CPU tests compare uninterrupted and interrupted/resumed runs
with AdamW, shuffled minibatches, dropout, and the same total epoch horizon:
model/optimizer tensors and validation NLL match exactly. CUDA RNG state is saved
too, but bitwise reproducibility across different hardware/compiler versions is
not claimed. Resume rejects changed config, data/preprocessing, or recorded
dependency versions. Compilation remains enabled and may warm up again.

`metrics.jsonl` is repaired from the committed checkpoint if preemption interrupts
its append. Existing epoch records are checked rather than silently duplicated
or discarded. In the rare case that a record must be recovered before its
checkpoint I/O duration was measured, that epoch's timing fields are null;
validation/train metrics remain available. ETA calculations skip missing durations
while retaining the correct completed-epoch count. Cumulative timing includes
recorded committed work and new recovery setup, not queue time or unrecorded
work lost to preemption. Original `config.json` remains; each recovery writes a
`resume-NNNN.json` runtime record.

```sh
# Same config, manifest, output and overrides as the original run:
.venv/bin/python train.py fit --config configs/dino-gaussian.toml \
  --manifest MANIFEST --output EXISTING_RUN --resume \
  --set selection_metric=calorie_nll --set wandb_mode=offline
```

## Optuna and Slurm

The serial controller resumes the existing SQLite study and reconnects to its
RUNNING trial before asking for the next queued proposal. Completed/pruned trials
are not trained again. Intermediate best calorie-NLL values and pruning decisions
are durable. A prior promotion decision is reused even if new competitors have
arrived. An ambiguous kill between Optuna's internal rung write and the explicit
decision write stops for inspection instead of silently changing the decision.
A process lock prevents two controllers from owning the same study.

```sh
.venv/bin/python sweep.py run PLAN BACKBONE MANIFEST EXISTING_STUDY --resume
```

Checkpoint-aware Babel submissions use `--requeue --open-mode=append`. When Slurm
requeues a preempted job, the batch script detects the persisted plan, resumes
the study, and restores the interrupted trial. It retains the same job/run/trial
IDs and uses the same W&B group. The local epoch relay continues from the same
W&B history. Ordinary code failures are marked failed and stop the controller;
they do not silently change settings or create an automatic retry loop. Requeue
is a scheduler action governed by cluster policy, and the two-hour allocation
limit is unchanged. See [Slurm preemption](https://slurm.schedmd.com/preempt.html).

## Existing experiments

Older checkpoints contain model weights and evaluation metadata only. They cannot
restore optimizer/RNG state and are explicitly rejected for faithful training
resume. Running processes do not acquire the new checkpoint format through a
source-file update; they must have started with this implementation. Existing
ResNet/Swin jobs are allowed to continue under their original trainer. The queued
DINO submission is replaced before startup with the checkpoint-aware version;
its old partial attempts remain archived, and its first new trial starts at epoch0.

Immutable sweep plans include source hashes. To adopt this implementation while
preserving every proposed setting, make an explicit source-provenance migration:

```sh
.venv/bin/python sweep.py refresh-plan ORIGINAL_PLAN CHECKPOINT_PLAN
```

This validates unchanged references and resolved trial configs, retains the
original protocol/proposals, and records the prior plan fingerprint/source hashes.
It cannot manufacture missing optimizer state for an old interrupted trial.
