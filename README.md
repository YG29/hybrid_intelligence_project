# Hybrid Intelligence Project
## Human Intervention Adaptive System for Classification Prediction

This project implements a hybrid intelligence system that combines:

- A fine-tuned **ResNet-18 classifier**, and  
- A **delegation model** that decides when to rely on the model vs. request human assistance.

The repository contains code for:

- Model training and fine-tuning  
- Human annotation and auditing  
- Automated and human-in-the-loop evaluation  

## 1. Setup

### Install dependencies

Create and activate a virtual environment, then install requirements:

```bash
pip install -r requirements.txt
```

## 2. Project Overview

```bash
.
├── notebooks/                      # Dataset exploration and overview notebooks
├── src/
│   ├── annotation_app.py           # Human annotation app for creating training data
│   ├── configure.py                # Paths and variables
│   └── utils.py                    # Utilities
├── test_and_evaluation/
│   ├── test_eval.py                # Automated testing and delegation inference
│   ├── test_annotation_app.py      # Human audit interface
│   └── result_eval.ipynb           # Final metric computation and analysis
├── finetune_resnet18.py            # Fine-tune ResNet-18 classifier
├── train_delegation.py             # Train delegation classifier
├── gendata.ipynb                   # Sample data for human annotation
├── human_annotation_analysis.ipynb # Analysis for human annotation data
├── requirements.txt
└── README.md
```

## 3. Workflow

A standard workflow:

1. Explore data in `notebook/`
2. Finetune Resnet
3. Sample annotation data
4. Annotate with `annotation_app.py`
5. Train delegation model
6. Run automated testing
7. Audit with `test_annotation_app.py`
8. Compute final metrics in `result_eval.ipynb`