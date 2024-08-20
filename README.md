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

## Requirements

- Python 3.11.9
- PyTorch 2.1.0
- NVIDIA GPU (experiments conducted on A40 48GB)

## Usage

Detailed instructions for running the experiments are provided in the code repository.

## Results

GADT3 outperforms state-of-the-art baselines in both single-domain and cross-domain settings, achieving average performance improvements of over 7% in AUROC and 16% in AUPRC.
