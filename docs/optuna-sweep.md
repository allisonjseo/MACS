# Frozen Gaussian head Optuna pilot

The first sweep runs 12 trials independently for each of ResNet-152, Swin V2,
and DINOv3: 36 trials total. Trial zero repeats the existing hyperparameters
at the full 30-epoch budget. Eleven proposals come from Optuna's seeded
`RandomSampler(seed=20261029)`. All backbones receive the identical proposal
design. This small pilot favors broad, matched exploration over fitting an
adaptive sampler from only a few completed trials. It does not use TPE.

## Search space and rationale

| Parameter | Reference | Search distribution | Motivation |
| --- | --- | --- | --- |
| Head LR | 0.001 | Log uniform [0.0001, 0.003] | Test slower, potentially more stable fits and modestly faster optimization. |
| AdamW weight decay | 0.01 | Log uniform [0.00001, 0.1] | Test regularization over several orders of magnitude. |
| Hidden width | 256 | Categorical {128, 256, 512} | Test head capacity without updating the encoder. |
| Dropout | 0.1 | Uniform [0, 0.4] | Test hidden-head regularization. |

These ranges are engineering choices centered around the existing reference,
not ranges inferred to be optimal. Existing validation curves suggest early
improvement followed by deterioration, especially for ResNet; this motivates
regularization and slower learning. The pilot cannot establish statistical
significance with one training seed. Changing hidden widths changes RNG
consumption at initialization, so a common seed does not imply identical
minibatch orders across architectures.

The objective and checkpoint selection both become **minimum calorie-only
validation Gaussian NLL, nats/example in physical kcal units**. Previously,
checkpoint selection minimized mean standardized NLL across five targets.
Training still minimizes the original equally weighted five-target Gaussian
NLL. Trial zero isolates the new selection rule while retaining reference
optimizer and head settings.

Every encoder stays pretrained and frozen, including BatchNorm and dropout.
Defaults retained from `configs/{resnet,swin,dino}-gaussian.toml`: 30 epochs,
batch 32, workers 4, seed 42, float32, AdamW betas (0.9, 0.999), eps 1e-8, gradient
clip 1, cosine horizon 30, Gaussian std floor 0.001 / initial std 1 in standardized
units, all five targets, no augmentation, training-only normalization, and
`torch.compile` Inductor/default/fullgraph/dynamic. Backbone LR 1e-5 and Smooth
L1 beta 1 remain inactive.

## Pruning criteria and evidence

Existing minimum calorie validation NLL occurred at epochs 9 (ResNet), 13 (Swin),
and 18 (DINO), with values 5.6913, 5.5006, 5.4229. At epoch 5 their best-so-far gaps
from those minima were 0.0159, 0.1996, 0.0980 nats; at epoch 10 they were 0, 0.0720,
0.0269. These three observations motivate a 10-epoch initial budget rather than
pruning at epoch 1 or 5. They do not establish that early ranking predicts final
ranking across unseen configurations.

Use `SuccessiveHalvingPruner(min_resource=10, reduction_factor=2,
min_early_stopping_rate=0, bootstrap_count=0)` separately per backbone. Report
**best-so-far** calorie validation NLL every epoch. Ask for pruning only after
epochs 10 and 20. Approximately the best half of trials that have reached a rung
are promoted; first arrivals have fewer competitors. This is asynchronous
successive halving, not exact cohort elimination. Trial zero is protected and
supplies reference scores at both rungs. Survivors finish 30 epochs. The cosine
schedule retains its 30-epoch horizon even for trials subsequently pruned.
There is no additional patience rule.

Best-so-far scoring matches selecting a saved checkpoint and avoids discarding
a trial only because its final rung epoch worsened. It can retain an early
lucky score. Slow learners can still be pruned unfairly. Study winners come
from **completed full-budget trials**; pruned trials retain their best
checkpoints and intermediate metrics for diagnosis.

See [Optuna's successive-halving documentation](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.SuccessiveHalvingPruner.html).

## Data, compute, and records

Use the same 2,204 training / 551 validation images, seed 42 split, manifest SHA256
`a940c2732a37b57ee4c0aed4dd5a4171c1834c410e2b491a3741d41bfcc4fbe1`.
No test data is loaded. Validation winners are subject to selection optimism;
a follow-up should repeat promising configurations on additional training
seeds before a final held-out test evaluation.

Each backbone gets one Babel Slurm job: one 48GB L40S, 4 CPUs, 16GB host RAM,
account `agu`, partition `preempt`, QoS `preempt_qos`, two-hour wall limit, no
automatic requeue. Trials are serial per study; studies can run concurrently.
Previous full fits took about 366, 520, 422 seconds. At those costs, 36 full fits
cost 4.36 GPU-hours. A rough estimate with pruning is 2–4 GPU-hours /45–90 minutes
elapsed with all GPUs available. Compilation, changed heads, checkpoint I/O,
queueing, and preemption can exceed that estimate; it is not a measured
speedup or confidence interval.

Each trial uses offline W&B, group `optuna-gaussian-pilot-20261029`, preserving
`train/`, `val/`, `timing/` namespaces. Credentials are not copied to Babel.
Completed and pruned runs finalize normal summaries and W&B records. Local
replay with `deploy/sync_babble.py` uploads them after fetching. SQLite databases,
CSV exports, source hashes, complete resolved configs, explicit overrides,
intermediate values, decisions, selected epochs and checkpoints stay on disk.
Hardware failures abort a study rather than silently adjusting settings.
After preemption, records persist but execution does not automatically resume;
inspect them and explicitly disclose any recovery launch.

## Reproduction

```sh
uv sync --locked --extra vision --extra analysis --extra sweep
# Planning does not launch training. Review the immutable JSON first.
.venv/bin/python sweep.py plan --spec configs/sweep-gaussian.toml \
  --output runs/optuna-pilot-plan.json
# On Babel after copying code and the reviewed plan:
bash deploy/sweep_babble.sh /home/cyrils/macs/data/training/manifest.json \
  /home/cyrils/macs/optuna-pilot-plan.json /home/cyrils/macs/optuna-pilot
```

The runner rejects source/config/data fingerprints or parameters that differ
from the plan, and refuses to overwrite an existing study. Per-backbone
`study-summary.json` reports the best completed trial and counts by Optuna
state; `trials.csv` includes pruned trials and intermediate scores.

The shared W&B group is `optuna-gaussian-pilot-20261029`. After fetching
the remote output tree into `runs/babel/optuna-pilot/`, publish each backbone's
completed/pruned histories and update the overview:

```sh
for backbone in resnet swin dino; do
  .venv/bin/python deploy/sync_babble.py "runs/babel/optuna-pilot/$backbone"
done
.venv/bin/python deploy/publish_sweep.py runs/optuna-pilot-plan.json \
  runs/babel/optuna-pilot
```

Run names include the backbone and trial number; tags identify backbone,
Optuna, and complete/pruned status. The overview table includes proposed
parameters, fetched status, selected NLL/epoch, elapsed time, and run links.
It is a snapshot of fetched records, not a live Slurm dashboard. The plan is
stored as a versioned W&B artifact; rerunning the publisher updates the same
overview run.

For automatic uploads while the submitted jobs run, use the bounded local relay:

```sh
.venv/bin/python deploy/watch_sweep.py runs/optuna-pilot-plan.json \
  runs/babel/optuna-pilot JOB_ID_RESNET JOB_ID_SWIN JOB_ID_DINO
```

It fetches JSON/CSV/log records every minute, skips checkpoints and SQLite files,
uploads newly completed/pruned trials, and refreshes the overview when fetched
configs or summaries change. It stops after the jobs leave the queue (with a final
fetch), or at a three-hour limit. It never launches or restarts training. Credentials
stay on the local machine. The relay requires the local process and network to
remain available; its snapshots are not continuous epoch-level online logging.
