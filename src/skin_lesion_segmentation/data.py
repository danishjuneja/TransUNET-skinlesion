"""Dataset discovery, deterministic splitting, and TensorFlow input pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import re
from typing import Iterable, Sequence


_IMAGE_ID = re.compile(r"(ISIC_\d{7})", re.IGNORECASE)


def is_augmented_path(path: Path) -> bool:
    """Identify augmentation files created beside the original ISIC records."""

    return any(tag in path.stem.lower() for tag in ("_flip_", "_saturated"))


@dataclass(frozen=True)
class Sample:
    """A paired dermoscopic image and binary segmentation mask."""

    image: Path
    mask: Path
    image_id: str


def extract_image_id(path: Path) -> str:
    """Return the canonical ISIC identifier embedded in a filename."""

    match = _IMAGE_ID.search(path.stem)
    if not match:
        raise ValueError(f"No ISIC identifier found in filename: {path.name}")
    return match.group(1).upper()


def discover_pairs(
    data_dir: Path | str,
    image_folder: str = "ISIC2018_Task1-2_Training_Input",
    mask_folder: str = "ISIC2018_Task1_Training_GroundTruth",
) -> list[Sample]:
    """Discover and pair inputs and masks by identifier, never list order."""

    root = Path(data_dir)
    image_dir = root / image_folder
    mask_dir = root / mask_folder
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(
            f"Expected image and mask folders under {root}. See data/README.md."
        )

    images: dict[str, Path] = {}
    for path in image_dir.glob("*.jpg"):
        if is_augmented_path(path):
            continue
        images[extract_image_id(path)] = path
    masks: dict[str, Path] = {}
    for path in mask_dir.glob("*.png"):
        # Ignore offline augmentation files left beside original masks.
        if is_augmented_path(path):
            continue
        masks[extract_image_id(path)] = path

    if not images:
        raise ValueError(f"No .jpg inputs found in {image_dir}")

    missing_masks = sorted(set(images) - set(masks))
    missing_images = sorted(set(masks) - set(images))
    if missing_masks or missing_images:
        details = []
        if missing_masks:
            details.append(f"missing masks for {missing_masks[:5]}")
        if missing_images:
            details.append(f"missing images for {missing_images[:5]}")
        raise ValueError("Unmatched dataset records: " + "; ".join(details))

    return [Sample(images[key], masks[key], key) for key in sorted(images)]


def split_samples(
    samples: Sequence[Sample],
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """Deterministically split already-paired samples into train/val/test sets."""

    if not 0 <= validation_fraction < 1 or not 0 <= test_fraction < 1:
        raise ValueError("Split fractions must be in [0, 1).")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must sum to less than 1.")
    if len(samples) < 3:
        raise ValueError("At least three paired samples are required.")

    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    n_total = len(shuffled)
    n_test = max(1, round(n_total * test_fraction))
    n_validation = max(1, round(n_total * validation_fraction))
    if n_test + n_validation >= n_total:
        raise ValueError("The dataset is too small for the requested split fractions.")

    test = shuffled[:n_test]
    validation = shuffled[n_test : n_test + n_validation]
    train = shuffled[n_test + n_validation :]
    return train, validation, test


def make_dataset(
    samples: Iterable[Sample],
    image_size: int = 256,
    batch_size: int = 4,
    training: bool = False,
    seed: int = 42,
):
    """Build a TensorFlow dataset with safe image/mask interpolation choices."""

    import tensorflow as tf

    records = list(samples)
    image_paths = [str(record.image) for record in records]
    mask_paths = [str(record.mask) for record in records]

    def load_pair(image_path, mask_path):
        image = tf.io.decode_jpeg(tf.io.read_file(image_path), channels=3)
        image = tf.image.resize(image, (image_size, image_size), method="bilinear")
        image = tf.cast(image, tf.float32) / 255.0

        mask = tf.io.decode_png(tf.io.read_file(mask_path), channels=1)
        mask = tf.image.resize(mask, (image_size, image_size), method="nearest")
        mask = tf.cast(mask >= 128, tf.float32)
        return image, mask

    def augment(image, mask):
        pair = tf.concat([image, mask], axis=-1)
        pair = tf.image.random_flip_left_right(pair, seed=seed)
        pair = tf.image.random_flip_up_down(pair, seed=seed + 1)
        return pair[..., :3], pair[..., 3:]

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    if training:
        dataset = dataset.shuffle(len(records), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.map(load_pair, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
