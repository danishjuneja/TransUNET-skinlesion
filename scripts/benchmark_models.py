"""Benchmark full U-Net and TransUNet architectures without training data."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

from skin_lesion_segmentation.models import build_transunet, build_unet

MEBIBYTE = 1024**2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure architecture complexity and CPU inference latency."
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/benchmark/cpu_architecture_benchmark.json"),
    )
    return parser.parse_args()


def percentile(values: list[float], percentage: float) -> float:
    """Calculate a linearly interpolated percentile without extra dependencies."""

    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def benchmark(model: tf.keras.Model, inputs: tf.Tensor, warmup: int, iterations: int):
    """Run synchronous batch-one inference and summarize wall-clock latency."""

    for _ in range(warmup):
        model(inputs, training=False).numpy()

    latency_ms = []
    for _ in range(iterations):
        start = time.perf_counter()
        model(inputs, training=False).numpy()
        latency_ms.append((time.perf_counter() - start) * 1000)

    total_parameters = model.count_params()
    trainable_parameters = sum(int(np.prod(weight.shape)) for weight in model.trainable_weights)
    return {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "non_trainable_parameters": total_parameters - trainable_parameters,
        "fp32_parameter_memory_mib": total_parameters * 4 / MEBIBYTE,
        "latency_ms": {
            "median": statistics.median(latency_ms),
            "mean": statistics.fmean(latency_ms),
            "p95": percentile(latency_ms, 95),
            "minimum": min(latency_ms),
            "maximum": max(latency_ms),
        },
        "throughput_fps_from_median": 1000 / statistics.median(latency_ms),
    }


def main():
    args = parse_args()
    if args.warmup < 1 or args.iterations < 2:
        raise ValueError("Use at least one warm-up and two measured iterations.")
    if args.threads < 1:
        raise ValueError("--threads must be at least one.")

    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.random.set_seed(42)
    inputs = tf.random.uniform((1, args.image_size, args.image_size, 3), seed=42)

    builders = {
        "U-Net": lambda: build_unet(image_size=args.image_size),
        "TransUNet": lambda: build_transunet(
            image_size=args.image_size,
            pretrained=False,
            freeze_backbone=False,
        ),
    }
    results = {}
    for name, builder in builders.items():
        print(f"Building and benchmarking {name}...")
        model = builder()
        results[name] = benchmark(model, inputs, args.warmup, args.iterations)
        tf.keras.backend.clear_session()

    payload = {
        "benchmark_type": "architecture-only synthetic-input CPU benchmark",
        "trained_weights_used": False,
        "dataset_used": False,
        "input_shape": [1, args.image_size, args.image_size, 3],
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "tensorflow_version": tf.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "tensorflow_intra_op_threads": args.threads,
        "tensorflow_inter_op_threads": 1,
        "models": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
