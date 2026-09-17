"""Train U-Net or TransUNet on paired ISIC-style data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from skin_lesion_segmentation.data import discover_pairs, make_dataset, split_samples
from skin_lesion_segmentation.metrics import dice_coefficient, dice_loss, soft_iou
from skin_lesion_segmentation.models import build_transunet, build_unet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("unet", "transunet"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    samples = discover_pairs(args.data_dir)
    train, validation, test = split_samples(samples, seed=args.seed)
    print(f"Paired samples: train={len(train)} validation={len(validation)} test={len(test)}")
    train_data = make_dataset(train, args.image_size, args.batch_size, training=True)
    validation_data = make_dataset(validation, args.image_size, args.batch_size)

    if args.model == "unet":
        model = build_unet(args.image_size)
    else:
        model = build_transunet(
            args.image_size,
            pretrained=not args.no_pretrained,
            freeze_backbone=not args.no_pretrained,
        )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss=dice_loss,
        metrics=[
            dice_coefficient,
            soft_iou,
            tf.keras.metrics.Recall(),
            tf.keras.metrics.Precision(),
        ],
    )
    run_dir = args.output_dir / args.model
    run_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            run_dir / "best.keras", monitor="val_loss", save_best_only=True
        ),
        tf.keras.callbacks.CSVLogger(run_dir / "history.csv"),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=5, min_lr=1e-7
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=12, restore_best_weights=True
        ),
    ]
    model.fit(
        train_data,
        validation_data=validation_data,
        epochs=args.epochs,
        callbacks=callbacks,
    )


if __name__ == "__main__":
    main()
