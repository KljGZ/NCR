# NCR: Neuron-Level Critical Replacement for Backdoor Attacks in Federated Learning

Research code for the paper "NCR: Neuron-Level Critical Replacement for Backdoor Attacks in Federated Learning".

## Overview
Federated learning (FL) is widely adopted in Artificial Intelligence of Things (AIoT) to address privacy and resource constraints; however, it is vulnerable to stealthy backdoor attacks due to heterogeneous device data distributions. Existing backdoor attacks typically involve uploading the entire maliciously optimized model, overlooking the critical attack surface that is determined by only a few key neurons, which significantly impact the effectiveness of an attack. Consequently, despite ongoing optimization efforts, conventional approaches still introduce substantial redundant parameters, increasing the risk of detection by defense mechanisms. Motivated by this observation, we propose a neuron-level critical replacement (NCR) strategy, which precisely replaces neuron-level parameters between malicious and benign models at a significantly finer granularity than existing techniques. Furthermore, we introduce a parameter-focusing mechanism that confines attacks to a limited number of highly sensitive neurons and their associated weights, significantly reducing unnecessary parameter perturbations. Even under a low poisoning rate, a backdoor can still be successfully implanted in the presence of nine advanced defense mechanisms. Moreover, validation experiments on the Udacity self-driving simulation platform further confirm NCR’s real-world threat to AIoT systems that rely on precise, model-driven decisions.

![NCR Overview](utils/overview.jpg)

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
## Project Structure
- `main_NCR.py`: main training/attack entry point
- `attack/`: attack implementations and utilities
- `defense/`: defense implementations
- `models/`: model architectures
- `utils/`: argument parsing, sampling, and helpers
- `run_NCR_*.sh`: runnable experiment presets

