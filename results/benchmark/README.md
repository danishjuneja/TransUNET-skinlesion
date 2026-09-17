# Architecture benchmark

This directory records a data-free comparison of the full 256 x 256 U-Net and
TransUNet architectures. The benchmark uses untrained models and a deterministic
random input tensor with batch size one. This is valid for parameter counting
and raw inference timing because the same operations execute regardless of the
input image or learned weight values.

Run it from an installed project environment:

```bash
python scripts/benchmark_models.py --warmup 5 --iterations 20 --threads 4
```

`cpu_architecture_benchmark.json` records raw measurements and the
software/hardware context. It is an architecture-only development-machine
benchmark, not a medical-quality evaluation or a mobile/production performance
claim. Historical segmentation quality remains documented separately in
`results/historical`.
