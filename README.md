# GADT3: Cross-Domain Graph Anomaly Detection via Test-time Training

This repository contains the implementation of GADT3 (Graph Anomaly Detection with Test-time Training), a novel framework for cross-domain graph anomaly detection.

## Overview

GADT3 addresses the challenge of detecting anomalies in graph-structured data when the test data is out-of-distribution and from heterogeneous domains. Key features include:

- Test-time training adaptation framework
- Homophily-based affinity score for self-supervised learning
- Dataset-specific encoders for handling heterogeneous feature spaces

## Datasets

The experiments use four real-world datasets:
- Amazon
- Reddit
- Facebook 
- YelpChi
- YelpRes
- YelpHotel


## Requirements
- Python 3.11.9
- PyTorch 2.1.0
- NVIDIA GPU (experiments conducted on A40 48GB)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/GADT3.git
cd GADT3

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Data Preparation
1. Download the datasets from their respective sources
2. Place them in the `data/` directory
3. Run preprocessing:
```bash
python data/preprocess.py
```

### Training & Testing
1. Train on source domain:
```bash
python main.py --source amazon --target facebook
```

2. Adapt to target domain:
```bash
python main.py --mode test-time --source amazon --target facebook
```

## Project Structure
```
GADT3/
├── model/
│   ├── sage.py       # GraphSAGE implementation
│   ├── mlp.py        # MLP predictor
│   └── gadt3.py      # Main GADT3 model
├── utils/
│   ├── metrics.py    # Evaluation metrics
│   ├── loss.py       # Loss functions
│   └── helper.py     # Helper utilities
└── data/
    ├── loader.py     # Data loading utilities
    └── preprocess.py # Data preprocessing
```
