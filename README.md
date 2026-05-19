# ST-VGANN: Spatial-Topological Vision Graph Attention Neural Network
[![License: AGPL v3](https://img.shields.io/badge/License-AGPLv3-orange.svg)](https://www.gnu.org/licenses/agpl-3.0) [![Python 3.12](https://img.shields.io/badge/Python-3.12-44cc11.svg)](https://www.python.org/downloads/) [![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/) [![PyG](https://img.shields.io/badge/PyG-%23EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)](https://pyg.org/) [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-brightgreen.svg)](https://github.com/Sinan-Pourazari/Wildlife-Re-ID/graphs/commit-activity)
&#x20;&#x20;

**Official Repository for:**\
*Exploration of Unsupervised Segmentation as a Foundation for Vision Graph Neural Networks in Individual Animal Re-identification*\

---

## Overview

This repository contains the complete source code, ablation framework, and orchestration logic required to reproduce the **ST-VGANN** proof-of-concept architecture.

ST-VGANN models animal morphology as an irregular topological manifold by extracting superpixel-based texture primitives through an unsupervised Convolutional Autoencoder (CAE) and routing them through Graph Attention Networks (GATv2) for parameter-efficient wildlife re-identification.

---

#  Features

- **Topological Feature Extraction**\
  Replaces rigid grid patches with adaptive SEEDS superpixels to better preserve biological contours and local structures.

- **Lightweight Architecture**\
  Approximately 11.4M parameters, designed to train on consumer-grade GPUs such as the NVIDIA RTX 5070 (12 GB).

- **Dual-Headed Disentanglement**\
  Combines SubCenter ArcFace identity clustering with orthogonal regularization against species-level taxonomic leakage.

- **Extensive Ablation Suite**\
  Includes a fully automated, idempotent orchestration framework capable of reproducing 48 architectural and topological configurations.

---

# Hardware Requirements & Environment Setup

All experiments were validated on a consumer workstation. 

## Validated Hardware

| Component | Specification                 |
| --------- | ----------------------------- |
| OS        | Windows 11 / Ubuntu 22.04 LTS |
| GPU       | NVIDIA RTX 5070 (12 GB VRAM)  |
| RAM       | 32 GB DDR5                    |

---

#  Installation

## 1. Clone the Repository

```bash
git clone 
cd Wildlife-Re-ID
```

## 2. Create a Virtual Environment

Python 3.12+ is recommended.

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Ensure that the installed PyTorch version matches your local CUDA toolkit configuration.

---

# Reproducing the Paper Results

This repository is designed for full reproducibility of the ST-VGANN ablation study and AnimalCLEF 2026 competition results.

All training, evaluation, caching, and artifact generation are automated through Python orchestration scripts.

---

## Step 1 — Dataset Preparation

Download the official datasets provided by the AnimalCLEF 2026 organizers, including:

- LynxID2025
- SeaTurtleID2022
- SalamanderID2025
- WildlifeReID-10k

### WildlifeReID-10k Repository

[https://github.com/WildMeOrg/wildlife-datasets](https://github.com/WildMeOrg/wildlife-datasets)

Place the extracted image folders and metadata CSV files into:

```text
Wildlife-Re-ID/src/images/
├── animal-clef-2026/
└── reid-10k/
```

---

## Step 2 — Running a Single Training Run (`run_single.py`)

The specific parameters, datasets, and training configuration can be adjusted directly inside `run_single.py`.

### Run a Single Experiment

```bash
python src/ml_utils/run_single.py
```

---

## Step 3 — Reproducing the Full Ablation Study (`run_ablation.py`)

The ablation framework automatically reproduces the architectural and topological configurations used in the paper.

### Run the Ablation Pipeline

```bash
python src/ml_utils/run_ablation.py
```

### Before Execution

1. Open `run_single.py` or `run_ablation.py`
2. Configure the dataset and training parameters
3. Enable or disable the required pipeline stages

### Pipeline Responsibilities

The framework automatically handles:

- Dataset stratification
- Superpixel extraction
- LMDB graph caching
- GNN training
- Embedding generation
- Evaluation
- Metric logging
- Ablation orchestration

### Output

Single runs are written to:

```text
runs/run_name/
```

Ablation outputs are written to:

```text
runs/ablations/
```

The first execution additionally generates persistent LMDB graph caches to accelerate future epochs and reruns.

---

## Step 4: Generating Paper figures

After evaluation is complete, the repository can automatically generate the tables and figures used in the paper.

### Generate Analysis Reports

```bash
python src/visualization/main.py
```

### Output

This script:

- Parses all evaluation directories
- Extracts peak ARI scores
- Generates publication-ready plots and tables
- Creates comparison plots matching the paper visualizations

Artifacts are written to:

```text
runs/ablations/analysis_reports/
```

---

### Generate the Appendix Ablation Table

```bash
python generate_appendix.py
```

### Output

Produces:

```text
appendix_table_populated.tex
```

The generated LaTeX table contains:

- Native ARI scores
- Jaccard-reranked ARI scores
- Optimized evaluation metrics
- All 43 configurations
- All evaluation domains

---

## Step 5 — Generating Methodology Visualizations

To reproduce the geometric and topological methodology figures used in the paper:

```bash
python test_cae.py
```
---

# Repository Structure

```text
Wildlife-Re-ID/
└── src/
    ├── bg_tools.py
    ├── main.py
    ├── ml_utils/
    │   ├── ablate_err.py
    │   ├── clear_err_ablate.py
    │   ├── dataloader.py
    │   ├── dataset_splitter.py
    │   ├── embedding_clusterings.py
    │   ├── eval.py
    │   ├── extract_embeddings.py
    │   ├── gnn/
    │   │   ├── cae.py
    │   │   ├── generate_cutouts.py
    │   │   └── gnn.py
    │   ├── helper.py
    │   ├── loss_mining_tools.py
    │   ├── merge_datasets.py
    │   ├── run_ablation.py
    │   ├── run_single.py
    │   ├── test_cae.py
    │   └── train_test_prototype.py
    └── visualization/
        ├── config.py
        ├── data.py
        ├── latex_export.py
        ├── main.py
        ├── plots.py
        └── tables.py
```

---

# Citation

*Citation details will be made available upon acceptance and publication of the workshop proceedings.*

# Acknowledgments

- This work builds upon evaluation methodologies provided by the WildlifeDatasets Toolkit.
- Data was provided through the LifeCLEF 2026 / AnimalCLEF challenge organizers.

---

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. This means that the full source code for any modified versions or expansions of this network—including those hosted on backend cloud systems, web platforms, or remote APIs—must be made entirely open and available to the public. See the [LICENSE](LICENSE) file for the complete legal text.

> *“This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation...”*