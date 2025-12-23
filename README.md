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
