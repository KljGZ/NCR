# NCR: Neuron-Level Critical Replacement for Backdoor Attacks in Federated Learning

Research code for the paper "NCR: Neuron-Level Critical Replacement for Backdoor Attacks in Federated Learning".

## Overview
Federated learning (FL) is widely adopted in AIoT to address privacy and resource constraints, but it remains vulnerable
to stealthy backdoor attacks due to heterogeneous data distributions. Existing attacks often upload an entire malicious
model, which introduces many redundant parameters and increases the chance of detection.

NCR (Neuron-Level Critical Replacement) targets this weakness by replacing only neuron-level parameters between
malicious and benign models, and by focusing updates on a small set of sensitive neurons and associated weights.
Experiments show NCR maintains high attack success rates (>= 90%) under multiple advanced defenses at low poisoning
rates, and validation on the Udacity self-driving simulator demonstrates practical risk.

## Requirements

Tested environment:
- torch==1.10.1+cu113
- torchvision==0.11.2+cu113
- torchaudio==0.10.1

### Option A: Build environment via script
```bash
bash enviorment.sh
```

### Option B: Install from requirements
```bash
pip install -r requirements.txt
```

Notes:
- Option A installs the CUDA 11.3 matching PyTorch build.
- If you already have a working PyTorch environment, Option B is usually enough.

## Quick Start
Create a directory to store results (and optional checkpoints):
```bash
mkdir -p save
```

Run NCR with a defense:
```bash
python main_NCR.py --model resnet --dataset cifar --attack NCR --lr 0.05 --defence flame
```

## Reproducible Runs
Predefined scripts run NCR across multiple defenses and datasets:
```bash
bash run_NCR_cifar.sh
bash run_NCR_fashion.sh
bash run_NCR_Mnist.sh
```

## Datasets
`main_NCR.py` supports `mnist`, `fashion_mnist`, and `cifar`. Datasets are downloaded via torchvision by default.
Non-IID client splits use `.npy` files in `data/` (for example, `data/iid_cifar.npy`).

## Key Arguments
Common flags (see `utils/options.py` for the full list):
- `--attack`: attack method (use `NCR` for this work)
- `--defence`: defense strategy (examples: `avg`, `flame`, `fltrust`, `multikrum`, `fld`, `alignins`, `snowball`, `scope`)
- `--dataset`: `mnist` | `fashion_mnist` | `cifar`
- `--model`: `cnn` | `resnet` | `VGG` | `mlp` (depending on dataset)
- `--poison_frac`: fraction of data to poison

## Output
Results are saved under the directory specified by `--save` (default: `save/`). The provided run scripts also create
`logs/` with per-run logs.

## Project Structure
- `main_NCR.py`: main training/attack entry point
- `attack/`: attack implementations and utilities
- `defense/`: defense implementations
- `models/`: model architectures
- `utils/`: argument parsing, sampling, and helpers
- `run_NCR_*.sh`: runnable experiment presets

## Citation
If you use this code in your research, please cite the paper.
