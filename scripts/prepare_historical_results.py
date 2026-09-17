"""Sanitize archived evaluation CSVs and generate comparison artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

METRICS = ("accuracy", "f1", "jaccard", "recall", "precision")


def read_archived(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        next(reader)
        for values in reader:
            image_name = Path(values[1].replace("\\", "/")).name
            metrics = dict(zip(METRICS, map(float, values[2:7])))
            rows.append({"image_id": Path(image_name).stem, **metrics})
    return rows


def write_rows(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image_id", *METRICS))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet", type=Path, required=True)
    parser.add_argument("--transunet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)

    models = {"U-Net": read_archived(args.unet), "TransUNet": read_archived(args.transunet)}
    summary = {}
    for model, rows in models.items():
        filename = model.lower().replace("-", "").replace(" ", "_") + "_per_image.csv"
        write_rows(args.output / filename, rows)
        means = {metric: float(np.mean([row[metric] for row in rows])) for metric in METRICS}
        summary[model] = {"n": len(rows), **means}
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    labels = ["Accuracy", "F1 / Dice", "Jaccard", "Recall", "Precision"]
    canvas = Image.new("RGB", (1600, 850), "#F8FAFC")
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("arialbd.ttf", 40)
        body_font = ImageFont.truetype("arial.ttf", 25)
        value_font = ImageFont.truetype("arialbd.ttf", 22)
    except OSError:
        title_font = body_font = value_font = ImageFont.load_default()
    draw.text(
        (800, 35),
        "Historical ISIC 2018 holdout evaluation (n = 518)",
        fill="#172033",
        font=title_font,
        anchor="ma",
    )
    left, top, right, bottom = 115, 120, 1530, 720
    for tick in range(0, 101, 20):
        y = bottom - tick / 100 * (bottom - top)
        draw.line((left, y, right, y), fill="#D8DEE9", width=2)
        draw.text((left - 18, y), str(tick), fill="#455268", font=body_font, anchor="rm")
    colors = ("#4F86C6", "#E4572E")
    group_width = (right - left) / len(METRICS)
    bar_width = 78
    for model_index, (_, values) in enumerate(summary.items()):
        for metric_index, metric in enumerate(METRICS):
            value = values[metric] * 100
            center = left + (metric_index + 0.5) * group_width
            x0 = center + (model_index - 0.5) * bar_width - bar_width / 2
            x1 = x0 + bar_width
            y0 = bottom - value / 100 * (bottom - top)
            draw.rounded_rectangle((x0, y0, x1, bottom), radius=8, fill=colors[model_index])
            draw.text(
                ((x0 + x1) / 2, y0 - 10),
                f"{value:.1f}",
                fill="#172033",
                font=value_font,
                anchor="ms",
            )
    for index, label in enumerate(labels):
        x = left + (index + 0.5) * group_width
        draw.text((x, bottom + 24), label, fill="#172033", font=body_font, anchor="ma")
    draw.text(
        (left, top - 22),
        "Mean per-image score (%)",
        fill="#455268",
        font=body_font,
        anchor="ls",
    )
    for index, model in enumerate(summary):
        x = 650 + index * 260
        draw.rounded_rectangle((x, 785, x + 34, 819), radius=6, fill=colors[index])
        draw.text((x + 48, 802), model, fill="#172033", font=body_font, anchor="lm")
    canvas.save(args.figure)


if __name__ == "__main__":
    main()
