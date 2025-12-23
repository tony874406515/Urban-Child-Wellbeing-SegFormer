# A Value-Based Programming Framework for Enhancing Children’s Well-being in Urban Communities

This repository contains the official implementation of the semantic segmentation model (based on SegFormer-B4) and the analytical framework described in the paper: "A Value-Based Programming Framework for Enhancing Children’s Well-being in Urban Communities".

The study integrates semantic recognition of street-view imagery (SVI) with multi-stakeholder questionnaires to evaluate urban environments for children.

## Repository Contents
* **`config.json`**: Hyperparameters and training configurations used in the study.
* **`utils.py`**: Label mapping rules between ADE20K and the 10 specific research categories.
* **`dataset.py`**: Data preprocessing pipelines (resolution normalization, augmentation).
* **`train.py`**: The training script utilizing the SegFormer-B4 architecture.

## Requirements
To install the necessary dependencies, run:
```bash
pip install -r requirements.txt
```

## Data Availability Statement
To protect privacy and comply with the licensing terms of the map service provider, the raw Street View Imagery (SVI) dataset used in this study is not publicly available in this repository.

**Access to Data:** The street-view image dataset and the trained model weights will be made available upon reasonable request for academic research purposes. Researchers interested in accessing the data should contact the corresponding author. Access is subject to the signing of a Data Usage Agreement and compliance with local data protection regulations.

## Usage
1. Configuration
The model parameters (Learning rate, Batch size, Optimizer) are defined in config.json following the experimental setup described in the paper.

2. Training
To reproduce the training process:

```Bash
python train.py --config config.json
```

## citation
If you find this code or research useful, please cite our paper: TBD
