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
python main_NCR.py --model resnet --dataset cifar --attack NCR --lr 0.05 --defence snowball
```
## Project Structure
- `main_NCR.py`: main training/attack entry point
- `attack/`: attack implementations and utilities
- `defense/`: defense implementations
- `models/`: model architectures
- `utils/`: argument parsing, sampling, and helpers
- `run_NCR_*.sh`: runnable experiment presets


