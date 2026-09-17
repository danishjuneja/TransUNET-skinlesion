"""Generate synthetic ellipse masks and RGB textures for pipeline smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.samples < 3:
        raise ValueError("At least three samples are required.")
    rng = np.random.default_rng(args.seed)
    image_dir = args.output / "ISIC2018_Task1-2_Training_Input"
    mask_dir = args.output / "ISIC2018_Task1_Training_GroundTruth"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    size = args.image_size
    for index in range(args.samples):
        image_id = f"ISIC_{index:07d}"
        base = rng.normal(185, 15, (size, size, 3)).clip(0, 255).astype("uint8")
        mask_image = Image.new("L", (size, size), 0)
        mask_draw = ImageDraw.Draw(mask_image)
        center = (
            int(rng.integers(size // 3, 2 * size // 3)),
            int(rng.integers(size // 3, 2 * size // 3)),
        )
        axes = (
            int(rng.integers(size // 10, size // 4)),
            int(rng.integers(size // 10, size // 4)),
        )
        mask_draw.ellipse(
            (
                center[0] - axes[0],
                center[1] - axes[1],
                center[0] + axes[0],
                center[1] + axes[1],
            ),
            fill=255,
        )
        mask = np.asarray(mask_image)
        lesion_color = rng.integers(35, 110, 3, dtype="uint8")
        image = np.where(mask[..., None] > 0, lesion_color, base)
        image = Image.fromarray(image.astype("uint8")).filter(
            ImageFilter.GaussianBlur(radius=2)
        )
        image.save(image_dir / f"{image_id}.jpg", quality=90)
        mask_image.save(mask_dir / f"{image_id}_segmentation.png")
    print(f"Generated {args.samples} synthetic pairs in {args.output}")


if __name__ == "__main__":
    main()
