# Data directory layout

This note describes the files present in the current evaluation workspace. The
downloaded dataset, feature cache, manifests, and run outputs are local data; the
repository `.gitignore` excludes `data/` and generated run directories.

## Workspace tree

```text
data/                                  symlink to this machine's scratch storage
├── nutrition5k/                       local Nutrition5k source/cache
│   ├── dish_ids/
│   │   ├── dish_ids_all.txt
│   │   ├── dish_ids_cafe1.txt
│   │   ├── dish_ids_cafe2.txt
│   │   └── splits/
│   │       ├── rgb_train_ids.txt
│   │       ├── rgb_test_ids.txt
│   │       ├── depth_train_ids.txt
│   │       └── depth_test_ids.txt
│   ├── metadata/
│   │   ├── dish_metadata_cafe1.csv
│   │   ├── dish_metadata_cafe2.csv
│   │   └── ingredients_metadata.csv
│   └── imagery/
│       └── realsense_overhead/
│           └── dish_<id>/rgb.png
└── nutrition5k-sift-full/              prepared full RGB test/reference set
    ├── manifest.json
    └── sift-features/
        └── <image-sha256>-f500.npy
```

## Raw source contents

The `nutrition5k` directory contains the official ID lists, metadata CSVs, and
overhead RGB photographs arranged by dish ID. The current ID lists contain 4,059
RGB training IDs and 709 RGB test IDs. Some IDs have no labels or no overhead
RGB image, so these counts are larger than the usable image counts in the SIFT
manifest. The depth split lists are present for reference; this image evaluation
uses RGB overhead photos.

The evaluator reads dish labels from the two cafe metadata files. It uses
`imagery/realsense_overhead/<dish-id>/rgb.png` as the image path when preparing
from this raw local layout. `ingredients_metadata.csv` is present in the source
directory but is not needed by the current image-level predictors.

## Prepared SIFT data

`nutrition5k-sift-full/manifest.json` records the selected test images, labeled
training references, label means, checksums, seed, and IDs whose overhead image
was missing. The current manifest has seed 42, all 507 available RGB test
overhead images, and all 2,755 available labeled RGB training references. It
records 202 missing test overhead images and 1,303 missing training-reference
overhead images among its candidates.

The manifest's `samples` are the held-out test set; its `references` are used by
SIFT retrieval. The test and reference IDs are required to be disjoint. In this
local-raw-data preparation, manifest image fields contain absolute paths into
the raw `nutrition5k/imagery` tree. Moving the scratch dataset can therefore
invalidate those paths even though the manifest and image checksums remain.

The SIFT feature cache currently contains 3,262 `.npy` files: descriptors for
the 2,755 references and 507 test samples. Filenames combine the image byte
SHA-256 and the feature limit (`f500`); this lets the evaluator reuse features
when the image content and SIFT limit are unchanged. The cache contains image
descriptors, not labels or predictions.

## Reproduce the SIFT evaluation

The repository does not need checked-in prediction or result directories. Start
with the raw source tree, prepare a manifest and feature cache under
`data/nutrition5k-sift-full`, then direct run output outside the repository (for
example, to `/tmp`). The detailed commands are in [`sift.md`](sift.md).

Preparation refuses to replace an existing `manifest.json`. To prepare again,
choose another `--data-dir`, or reuse the existing manifest and cache for runs.
