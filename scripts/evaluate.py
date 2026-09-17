"""Evaluate a saved segmentation model with per-image pixel metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
import tensorflow as tf

from skin_lesion_segmentation.data import discover_pairs, split_samples
from skin_lesion_segmentation.metrics import dice_coefficient, dice_loss, soft_iou
from skin_lesion_segmentation.models.transunet import AddPositionEmbeddings, TransformerBlock


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("unet", "transunet"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation.csv"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    _, _, test = split_samples(discover_pairs(args.data_dir), seed=args.seed)
    custom_objects = {
        "dice_coefficient": dice_coefficient,
        "dice_loss": dice_loss,
        "soft_iou": soft_iou,
        "AddPositionEmbeddings": AddPositionEmbeddings,
        "TransformerBlock": TransformerBlock,
    }
    model = tf.keras.models.load_model(args.weights, custom_objects=custom_objects)

    rows = []
    for sample in test:
        image = cv2.cvtColor(cv2.imread(str(sample.image)), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (args.image_size, args.image_size)).astype("float32") / 255.0
        mask = cv2.imread(str(sample.mask), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (args.image_size, args.image_size), interpolation=cv2.INTER_NEAREST)
        expected = (mask >= 128).astype("uint8").ravel()
        predicted = (model.predict(image[None, ...], verbose=0)[0, ..., 0] >= args.threshold)
        predicted = predicted.astype("uint8").ravel()
        rows.append(
            {
                "image_id": sample.image_id,
                "accuracy": accuracy_score(expected, predicted),
                "f1": f1_score(expected, predicted, zero_division=0),
                "jaccard": jaccard_score(expected, predicted, zero_division=0),
                "recall": recall_score(expected, predicted, zero_division=0),
                "precision": precision_score(expected, predicted, zero_division=0),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print({key: float(np.mean([row[key] for row in rows])) for key in rows[0] if key != "image_id"})


if __name__ == "__main__":
    main()

