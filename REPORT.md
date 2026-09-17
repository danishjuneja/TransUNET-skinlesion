# Technical Report: U-Net vs. TransUNet for Skin-Lesion Segmentation

## Executive summary

This project compares U-Net with a hybrid TransUNet architecture on binary
lesion-boundary segmentation using ISIC 2018 images. Archived per-image
evaluation records show that TransUNet improved mean recall from 0.7729 to
0.9442 on a 518-image holdout set. It also improved mean F1 by 0.0740 and mean
Jaccard score by 0.0776, while precision decreased by 0.0510.

The result is promising but should be presented as a historical project result,
not as a newly reproduced benchmark. The original model checkpoints and exact
input-image snapshot are not part of this cleaned repository.

## Problem statement

Skin-lesion segmentation separates the lesion foreground from surrounding skin
in a dermoscopic image. U-Net is effective at recovering local boundaries, but
convolutions aggregate context locally. TransUNet adds self-attention over
tokenized CNN features so distant regions can interact while a U-shaped decoder
retains high-resolution localization cues.

## Dataset and split

The original experiment targeted Task 1 of the ISIC 2018 challenge. The
archived code resized RGB inputs and single-channel masks to 256 x 256 pixels,
normalized them to `[0, 1]`, and used deterministic train/validation/test splits
with seed 42. The intended split was approximately 60/20/20.

The cleaned loader improves data integrity by matching each input with its mask
using the ISIC identifier before any split. This avoids relying on the ordering
of independent filename lists. It also uses nearest-neighbor mask resizing to
preserve binary labels.

## Models

### U-Net baseline

The baseline has four encoder stages with 64, 128, 256, and 512 filters, a
1,024-filter bridge, transposed-convolution upsampling, skip concatenation, and
a one-channel sigmoid output.

### TransUNet

The hybrid model uses ImageNet-pretrained ResNet50V2 features, a learned linear
token projection, learned positional embeddings, 12 transformer blocks with 12
attention heads, and a four-stage convolutional decoder. Three ResNet feature
maps are provided to the decoder as spatial skip connections.

## Training protocol recovered from the archive

| Setting | Value |
|---|---|
| Input size | 256 x 256 RGB |
| Batch size | 4 |
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Loss | Soft Dice loss |
| Archived epochs | 5 |
| Seed | 42 |
| Decision threshold | 0.5 |

The current scripts expose these choices as command-line arguments and save the
best validation-loss checkpoint. Training-time metrics include soft Dice, IoU,
recall, and precision.

## Historical evaluation

Metrics below are arithmetic means of 518 per-image records in the archived
CSV files. The sanitized records retain only image identifiers and numeric
metrics; old absolute user paths have been removed.

| Model | Accuracy | F1 | Jaccard | Recall | Precision |
|---|---:|---:|---:|---:|---:|
| U-Net | 0.9241 | 0.7911 | 0.7052 | 0.7729 | 0.8842 |
| TransUNet | 0.9468 | 0.8651 | 0.7828 | 0.9442 | 0.8331 |

TransUNet's higher recall indicates fewer false-negative lesion pixels. The
lower precision indicates more false-positive pixels. For an application where
missing a lesion region is more costly than modest over-segmentation, this
trade-off may be desirable; any clinical interpretation would still require a
properly controlled validation study.

## Architecture complexity and inference cost

A separate data-free benchmark compared the full models using an untrained
batch-one synthetic tensor at 256 x 256. On a Windows development machine with
16 logical CPUs, TensorFlow 2.15.1 was restricted to four intra-op threads and
one inter-op thread. Five passes were used for warm-up and 20 for measurement.

| Model | Parameters | FP32 parameter memory | Median CPU latency | p95 latency |
|---|---:|---:|---:|---:|
| U-Net | 31.05M | 118.44 MiB | 750.2 ms | 786.5 ms |
| TransUNet | 100.89M | 384.85 MiB | 1,149.1 ms | 1,208.2 ms |

TransUNet therefore required about 3.25 times the parameter memory and 1.53
times the median inference time in this setup. These measurements are useful
for comparing architecture cost, but they are not mobile-device or production
latency claims. They are also independent of the historical quality experiment.

## Engineering improvements in this repository

1. Replaced hard-coded Windows paths with CLI arguments and `pathlib` paths.
2. Centralized deterministic pairing and splitting of image-mask records.
3. Added input validation and actionable errors for missing or unmatched data.
4. Removed the unnecessary TensorFlow Addons dependency.
5. Added safe binary-mask resizing and on-the-fly paired augmentation.
6. Added small configurable TransUNet variants for resource-aware smoke tests.
7. Added tests that verify dataset pairing and guard the historical summaries.
8. Added a synthetic generator for pipeline testing without medical data.
9. Added Git hygiene for raw data, weights, logs, predictions, and personal PDFs.
10. Preserved third-party attribution and separated claims from provenance.

## Limitations

- The original inputs and checkpoints are unavailable in the distributable
  repository, so the historical metrics cannot be regenerated here as-is.
- Five epochs are insufficient for a definitive architecture comparison.
- No confidence intervals or multi-seed analysis were recorded.
- The historical rows are per-image pixel metrics, not official ISIC leaderboard
  submissions, and should not be described as challenge test-set scores.
- Accuracy is inflated by background pixels and is less informative than Dice,
  Jaccard, recall, and precision for lesion segmentation.
- The default TransUNet is compute-heavy and has not been optimized for XR/mobile
  deployment.

## Suggested XR/edge continuation

For an XR-focused extension, distill the transformer model into a mobile U-Net,
replace the ResNet50V2 backbone with MobileNetV3 or EfficientNet-Lite, benchmark
latency and memory at 128/192/256-pixel resolutions, and export a quantized
TensorFlow Lite model. Report quality-latency-Power trade-offs on target hardware
rather than accuracy alone.

## Claim-safe résumé wording

> Implemented TensorFlow U-Net and hybrid TransUNet models for ISIC 2018 skin-
> lesion segmentation; archived holdout evaluation improved mean pixel recall
> from 77.3% to 94.4% (+17.1 percentage points), with reproducible data-pairing,
> evaluation, and result-provenance tooling.
