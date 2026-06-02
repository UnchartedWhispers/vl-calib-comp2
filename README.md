# VL-Calibration Comp2 Project

This repository sets up a reproducible local development environment for experimenting with the VL-Calibration dataset and preparing GPU-side experiments for the Computing for Data Science 2 final project.

The current goal is to do as much development as possible locally, then use the GPU server only for GPU-dependent execution, profiling, benchmarking, and final analysis.

## Project Goal

This project explores the VL-Calibration pipeline around decoupled visual and reasoning confidence for large vision-language models. The immediate engineering goal is to build a clean, reproducible pipeline that can:

1. Download and organize the VL-Calibration-12K dataset.
2. Verify the raw dataset schema.
3. Convert the raw dataset into a pipeline-friendly format.
4. Run local smoke tests without requiring a GPU.
5. Recreate the same environment later on a GPU server.
6. Extend the pipeline to generate model responses, visual entropy, visual KL, and related fields required by later training or analysis scripts.

## Repository Structure

```text
vl-calib-comp2/
  configs/
  data/
    raw/
    debug/
    processed/
    modelscope/
  logs/
  results/
  scripts/
    check_env.py
    check_schema.py
    download_vl_calibration_cli.sh
    export_parquet_to_jsonl.py
    make_debug_subset.py
    prepare_raw_for_pipeline.py
    smoke_test_local.sh
  src/
  environment.yml
  environment.lock.yml
  requirements.lock.txt
  README.md
```

## Data Layout

The raw ModelScope dataset contains the following fields:

```text
problem
answer
images
```

The later VL-Calibration pipeline expects additional fields such as:

```text
response
ground_truth
vision_entropy
vision_kl
vision_token_count
vision_kl_token_count
```

These extra fields are not present in the raw dataset. They should be produced later by a model generation and visual-certainty computation stage.

The local preprocessing step currently creates placeholder fields so the downstream schema can be debugged before GPU access.

## Environment Setup

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate vl-calib-comp2
```

Check that the environment works:

```bash
python scripts/check_env.py
```

After changing packages, update the lock files:

```bash
pip freeze > requirements.lock.txt
conda env export --no-builds > environment.lock.yml
```

## Dataset Download

The preferred download path is the ModelScope CLI, not `MsDataset.load()`.

The Python `MsDataset.load()` path caused version conflicts between `modelscope` and Hugging Face `datasets`, so this repository uses a CLI-based download script instead.

Run:

```bash
bash scripts/download_vl_calibration_cli.sh
```

This downloads the dataset into:

```text
data/modelscope/VL-Calibration-12K/
```

Then inspect the downloaded files:

```bash
find data/modelscope/VL-Calibration-12K -maxdepth 5 -type f | sort
```

If the dataset contains a `train.jsonl`, copy it to the expected raw-data path:

```bash
mkdir -p data/raw/VL-Calibration-12K
cp path/to/train.jsonl data/raw/VL-Calibration-12K/train.jsonl
```

If the dataset contains a Parquet file instead, convert it:

```bash
python scripts/export_parquet_to_jsonl.py \
  --input path/to/train.parquet \
  --output data/raw/VL-Calibration-12K/train.jsonl
```

## Local Smoke Test

After the raw dataset exists at:

```text
data/raw/VL-Calibration-12K/train.jsonl
```

run:

```bash
bash scripts/smoke_test_local.sh
```

The smoke test performs the following steps:

1. Checks the Python environment.
2. Creates a 32-example debug subset.
3. Checks the raw dataset schema.
4. Converts the raw examples into a pipeline-friendly format.
5. Checks the converted schema.
6. Confirms that the local pipeline runs end-to-end.

Expected final message:

```text
Local smoke test passed.
```

## Current Status

Done:

* Created a reproducible conda environment.
* Added environment checking script.
* Added CLI-based ModelScope dataset downloader.
* Avoided the unstable `MsDataset.load()` path.
* Added raw dataset schema checker.
* Added debug subset creation script.
* Added raw-to-pipeline format conversion script.
* Added local smoke test script.
* Confirmed that the local smoke test passes.
* Committed the working local setup.

## Important Notes

The converted local file may contain fields such as:

```text
response
vision_entropy
vision_kl
vision_token_count
vision_kl_token_count
```

but these are placeholders at the local preprocessing stage.

A field existing in the JSONL file does not mean it contains real model-generated values yet. The real values should be computed later on the GPU server or with a model inference pipeline.

The local machine is used for correctness, structure, and pipeline debugging. The GPU server will be used for model execution, CUDA/PyTorch profiling, benchmarking, and final measurements.

## Reproducing on a GPU Server

On the GPU server:

```bash
git clone <repo-url>
cd vl-calib-comp2

conda env create -f environment.yml
conda activate vl-calib-comp2

bash scripts/download_vl_calibration_cli.sh
bash scripts/smoke_test_local.sh
```

If PyTorch/CUDA is needed later, install the correct PyTorch build for the GPU server separately according to the server CUDA version.

## To-Do List

Near-term:

* Inspect `decouple.py` and identify exactly which fields it reads.
* Decide whether to keep `decouple.py` unchanged and write a preprocessing script, or modify `decouple.py` to accept the raw dataset format.
* Write a small script that fills `response` using a tiny model or mock response for local debugging.
* Add validation checks that fail if required fields are missing or still `None` before GPU-only stages.
* Add a README section explaining each script and its inputs/outputs.

GPU-side:

* Recreate the environment on the GPU server.
* Install the correct PyTorch/CUDA stack.
* Run the local smoke test on the GPU server.
* Run model inference on a small subset.
* Generate real `response` fields.
* Compute or approximate visual entropy and visual KL fields.
* Verify compatibility with `decouple.py`.
* Add benchmark scripts for timing and profiling.
* Collect runtime, memory, and profiling results.
* Save results in `results/`.

Report and presentation:

* Document the full data pipeline.
* Record exact system configuration.
* Record package versions and GPU information.
* Compare baseline and optimized versions.
* Include runtime and profiling results.
* Prepare final plots for speedup, bottlenecks, and GPU utilization.
* Summarize limitations and next steps.