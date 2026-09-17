# Data

Raw medical images are intentionally not versioned in this repository.

For a real run, download Task 1 training images and ground-truth masks from the
[ISIC 2018 challenge](https://challenge.isic-archive.com/landing/2018/45/) and
place the extracted folders under `data/isic2018/` as shown in the root README.
Review the dataset's current terms and citation instructions before use.

For a code-path smoke test with no medical data, run:

```bash
python scripts/generate_synthetic_data.py --output data/synthetic --samples 24
```

Synthetic examples are procedurally generated and must never be mixed with or
reported as benchmark results.

