# Image regression training

Three pretrained encoders share the same training pipeline and five **independent**
regression heads: `calories_kcal`, `protein_g`, `carbs_g`, `fat_g`, and `mass_g`.
Each head is Linear(features, 256), ReLU, Dropout(0.1), Linear(256, 1).
Use `--set hidden_dim=0` for independent linear heads.

| Config | Exact timm pretrained model | Input |
| --- | --- | --- |
| [`resnet.toml`](../configs/resnet.toml) | `resnet152.a1_in1k` | 224×224 |
| [`swin.toml`](../configs/swin.toml) | `swinv2_base_window12to16_192to256.ms_in22k_ft_in1k` | 256×256 |
| [`dino.toml`](../configs/dino.toml) | `vit_base_patch16_dinov3.lvd1689m` | 256×256 |

These are ResNet-152 with the A1 training recipe, Swin **V2**, and **DINOv3**
ViT-B/16. ResNet has no single version called “latest”; the exact checkpoint is
named here to make the comparison unambiguous. Sources:
[ResNet](https://huggingface.co/timm/resnet152.a1_in1k),
[Swin V2](https://huggingface.co/timm/swinv2_base_window12to16_192to256.ms_in22k_ft_in1k),
[DINOv3](https://huggingface.co/timm/vit_base_patch16_dinov3.lvd1689m).
Respect the pretrained checkpoints' respective licenses, including DINOv3's.

The reference configs differ **only** in `backbone`. All use pretrained weights,
a frozen encoder (including BatchNorm statistics), the heads above, seed 42,
30 epochs, batch size 32, four loader workers, one CUDA GPU, float32, AdamW
(head LR 0.001, backbone LR 0.00001 when unfrozen, weight decay 0.01,
betas 0.9/0.999, epsilon 1e-8), cosine decay to zero, gradient clipping at 1.0,
and Smooth L1 with beta 1 on standardized targets. `torch.compile` is enabled
with Inductor, default mode, fullgraph capture, and dynamic batch shapes; W&B
logging defaults to online in the `macs-nutrition5k` project. Each quantity has equal loss
weight after standardization. Target means and population standard deviations
come **only from the training partition**, with a 1e-6 standard deviation floor.
There is no early stopping, accumulation, batch dropping, augmentation, or
automatic change of device, batch size, or precision after an error.

Each backbone uses its own pretrained resize/center crop, bicubic interpolation,
and ImageNet normalization, for both training and inference. Center cropping
can remove peripheral food; evaluate this choice on validation data before
changing it. Predictions invert target standardization and clamp below zero.
The model does not estimate uncertainty.

## Local setup and data

```sh
bash deploy/setup.sh
uv run --locked --extra vision train.py prepare \
  --source-dir data/nutrition5k \
  --output data/nutrition5k-training/manifest.json \
  --seed 42 --validation-fraction 0.2
```

Preparation reuses the raw overhead image layout and existing `images/` or
`references/` caches. It downloads absent official split lists, metadata, and
**training** overhead photos, records 404s/missing labels, then uses a seeded
shuffle of sorted available official RGB training dish IDs for an 80/20 split
(validation count rounded up). It never loads test photographs or assigns test
labels to training/validation. Manifests validate disjoint IDs and record image
checksums, split seed, exclusions, and the official test IDs. Train and validation
use dish IDs; this does not establish independence of related physical meals.

Image paths are relative to the manifest directory. To move prepared data,
preserve its relative directory layout (including the source images), or prepare
it on the destination. Existing manifests and training run directories are never
overwritten. All three backbones should use the same manifest.

```sh
# Inspect fully resolved settings without downloading weights or starting training.
uv run --locked --extra vision train.py fit --config configs/resnet.toml \
  --manifest data/nutrition5k-training/manifest.json --output runs/resnet --dry-run

# Train the heads with the reference settings.
bash deploy/run.sh resnet data/nutrition5k-training/manifest.json runs/resnet

# Explicit alternative: fine-tune the entire DINOv3 encoder as well as the heads.
bash deploy/run.sh dino data/nutrition5k-training/manifest.json runs/dino-finetune \
  --set freeze_backbone=false --set head_lr=0.0001 --set backbone_lr=0.00001
```

Every invocation prints the config file, **all** resolved defaults, and differences
from that file before loading weights or performing optimizer steps. The first
training invocation downloads the pretrained checkpoint into the normal
Hugging Face/torch cache. Set `HF_HOME`/`TORCH_HOME` to persistent storage as
appropriate; preload weights if compute nodes cannot access the Internet.
No random-weight fallback occurs if loading pretrained weights fails.

Before an agent launches a real experiment, it must also show that exact config,
runtime overrides, and changes from the prior/reference run in chat, per the
working agreement. Script output alone is not a replacement for that agreement.

## Bridges-2 / PSC

The configured SSH alias is `bridges2` (`csharma2@bridges2.psc.edu`). A read-only
check confirmed `GPU-shared`, `h100-80`, and
`/ocean/projects/cis260267p/csharma2` on this account. The scripts follow PSC's
[GPU-shared batch instructions](https://www.psc.edu/resources/bridges-2/user-guide/#gpu-partitions).

```sh
# From the local checkout; copies code without data, runs, credentials or .venv.
bash deploy/sync.sh bridges2 /ocean/projects/cis260267p/csharma2/MACS
ssh bridges2
cd /ocean/projects/cis260267p/csharma2/MACS
bash deploy/setup.sh

# Prepare the training manifest on PSC (or transfer the source/cache first).
.venv/bin/python train.py prepare --source-dir data/nutrition5k \
  --output data/nutrition5k-training/manifest.json

# Set the allocation that has GPU SUs for your account.
export PSC_ACCOUNT=YOUR_GPU_ALLOCATION
bash deploy/bridges2.sh data/nutrition5k-training/manifest.json runs/psc
```

This submits array tasks 0/1/2 for ResNet/Swin/DINO respectively. Each gets one
H100-80 GPU, one task, eight CPUs, 32 GiB host memory, and a four-hour walltime
in `GPU-shared`. Logs go to `OUTPUT_ROOT/logs/JOB_TASK.{out,err}` and model
outputs to `OUTPUT_ROOT/ARRAY_JOB_ID-MODEL/`. Before submission, scheduler
settings and all three fully resolved training configs are printed. At runtime,
the final job-specific paths, hardware and training configs are printed again.

Resource overrides: `PARTITION`, `GPU_TYPE`, `CPUS`, `MEMORY`, `WALLTIME`.
Model overrides: append repeated `--set key=value` arguments. For example,
`WALLTIME=08:00:00 bash deploy/bridges2.sh MANIFEST OUTPUT --set epochs=60`.
These choices are intentional changes to the reference run. Changing to V100
may require a PyTorch/CUDA build with Volta support; verify the chosen environment
and disclose that environment change before launching. The supplied environment
uses `uv.lock` and does not load PSC's older PyTorch modules automatically.

## Babel (Babble)

The cluster connection is `cyrils@login.babel.cs.cmu.edu` with
`~/.ssh/id_ed25519`. It was found in existing deployment configurations and
verified over SSH. The launcher submits one Slurm job to `preempt`, account
`agu`, QoS `preempt_qos`, and runs all three models **serially on the same GPU**.
Defaults are one H200, four CPUs, 32 GiB host memory, four hours, no requeue.
Set `GPU_CONSTRAINT=L40S` to select an L40S instead; report hardware changes
alongside the training configuration before launch.

```sh
# On Babel, after staging the checkout and installing the environment:
.venv/bin/python train.py prepare --source-dir /home/cyrils/macs/data/nutrition5k \
  --output /home/cyrils/macs/data/training/manifest.json --download-workers 16
GPU_CONSTRAINT=L40S bash deploy/babble.sh \
  /home/cyrils/macs/data/training/manifest.json /home/cyrils/macs/runs \
  --set wandb_mode=offline \
  --set wandb_entity=sharma-cyril-carnegie-mellon-university \
  --set wandb_group=babel-compiled-20261029
```

`BABEL_ACCOUNT`, `PARTITION`, `QOS`, `GPU_CONSTRAINT`, `CPUS`, `MEMORY`, and
`WALLTIME` override scheduler settings. `PYTHON` selects the training interpreter.
Outputs are `OUTPUT_ROOT/JOB_ID-MODEL`, logs are under `OUTPUT_ROOT/logs`,
and `JOB_ID-suite-summary.json` records wall-clock process time per model.
Caches live next to the staged checkout, under `cache/`. No credentials are
copied to the cluster by these scripts.

## W&B and runtime measurements

The project is
[macs-nutrition5k](https://forge.coreweave.com/wandb/sharma-cyril-carnegie-mellon-university/macs-nutrition5k).
`wandb_mode` is explicit (`online`, `offline`, or `disabled`); failures do not
silently disable logging. `wandb_project`, `wandb_entity`, and `wandb_group` are
recorded in the resolved config and can be overridden with `--set`. Each model
gets its own run with the full config. Loss and per-target MAE are grouped
under `train/` and `val/`, learning rates under `train/lr/`, and **all** authored
timing/throughput metrics and timing summaries under `timing/`.
Images and checkpoint weights are not uploaded automatically.

For Babel, use **offline** records and sync from the local machine so the W&B
key stays local:

```sh
# On this machine, fetch logs/metrics/W&B records (excludes large .pt files)
# and upload the completed JSON logs to their configured W&B project.
bash deploy/fetch_babble.sh
```

The fetch script accepts a local output directory (default `runs/babel`) and
`BABEL_HOST`, `BABEL_IDENTITY`, `BABEL_OUTPUT` overrides. Authenticate W&B locally
using the normal SDK environment or login. `wandb.json` records run identity;
`summary.json` contains setup time, total fit time, training-loop time, first
epoch duration, and mean duration of later epochs. Epoch metrics include
CUDA-synchronized train/validation durations, first-batch times and checkpoint
I/O time. The first epoch includes lazy compilation and warm-up; later epochs
provide a steady-state comparison. This does **not** isolate pure compiler time
from other first-epoch costs. `total_seconds` excludes W&B final synchronization;
the suite wall-clock timing includes the complete child process.

Named phases use [codetiming](https://github.com/realpython/codetiming), with CUDA
synchronization around measured device work. Every epoch reports
`timing/estimated_remaining_seconds` and `timing/estimated_total_seconds` based
on the median of the latest five completed epochs, excluding the first warm-up
epoch once another epoch exists. Low/high remaining-time estimates use the
minimum/maximum of those recent durations; they are heuristic ranges.

`fetch_babble.sh` replays the authoritative JSON logs through the local W&B SDK,
so even runs already in flight when reporting changed get consistent metric
sections and historical ETA estimates. Completed uploads are marked locally to
avoid duplicate uploads. Remote SDK system monitoring remains in the fetched
offline records; replay disables local-machine system monitoring to avoid
reporting laptop utilization as cluster utilization.

Compilation is in place (`nn.Module.compile`, which calls `torch.compile`),
so checkpoint state keys stay compatible with eager inference. With the default
`compile_fullgraph=true`, graph breaks raise errors. Disabling compilation,
changing its backend/mode, or changing precision must be an explicit, reported
configuration change. `compile_backend=eager` is used only in the tiny CPU test;
actual training uses Inductor.

## Artifacts and evaluation

Each training run writes:

- `config.json`: complete configuration, explicit overrides, preprocessing,
  manifest hash, exact training/validation IDs, normalization values, dependency
  versions, Python/Git provenance, deterministic settings and hardware.
- `metrics.jsonl`: train/validation standardized loss and MAE for all five
  quantities in their original units, plus learning rates, at each epoch.
- `best.pt`, `last.pt`: atomic model checkpoints, including encoder weights,
  heads, normalization buffers, config and preprocessing. `best.pt` is selected
  only by validation standardized loss. Checkpoint loading needs no weight download.
- `summary.json`: written on completion with the best validation loss.

The checkpoints support inference, not optimizer-state resume. After interruption,
`last.pt` is usable for inference; a fresh training run requires a new output
directory and an explicitly reported configuration.

```sh
# Prepare the same held-out benchmark used by the existing baselines.
uv run --locked --extra vision eval.py prepare --all --data-dir data/nutrition5k-test
uv run --locked --extra vision eval.py run --predictor vision \
  --checkpoint runs/resnet/best.pt --device cuda --data-dir data/nutrition5k-test
```

The evaluator scores calories/protein/carbs/fat through the existing `Prediction`
contract. The fifth head's `mass_g` estimate is saved in each prediction's
`backend_metadata`; its validation MAE is in training metrics. Official benchmark
mass MAE is not added to the existing four-target summary schema. Checkpoint
SHA-256, model identity, transform settings and vision dependency versions are
recorded in evaluation configs.

Offline verification:

```sh
uv run --locked --extra vision ruff check .
uv run --locked --extra vision ruff format --check .
uv run --locked --extra vision ty check
uv run --locked --extra vision pytest
```
