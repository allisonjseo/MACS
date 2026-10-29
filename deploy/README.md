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
and Smooth L1 with beta 1 on standardized targets. Each quantity has equal loss
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

## Babble

There was no `babble` entry in the local SSH config. Until its hostname and
scheduler are supplied, use the foreground launcher **on Babble**, inside a GPU
allocation if one is required:

```sh
# Substitute the actual Babble alias and remote project directory.
bash deploy/sync.sh YOUR_BABBLE_ALIAS /absolute/project/path/MACS
# On Babble:
bash deploy/setup.sh
.venv/bin/python train.py prepare --source-dir data/nutrition5k \
  --output data/nutrition5k-training/manifest.json
bash deploy/babble.sh data/nutrition5k-training/manifest.json runs/babble
```

This trains the three models serially on one GPU. `RUN_ID` optionally sets the
output prefix. `PYTHON` optionally selects an existing compatible interpreter
for either deployment launcher. Scheduler-specific Babble submission is not
configured without knowing that system's scheduler and resource names.

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
