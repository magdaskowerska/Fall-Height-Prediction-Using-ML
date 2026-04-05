# Fall Height Prediction Using Machine Learning

## Background

This work is part of a broader collaboration with the Department of Biomedical and Health Sciences, Forensic Medicine Unit of the University of Milan, which provided two specialized datasets on fatal falls: one collected in the metropolitan area of Milan and the other in Berlin. The datasets contain autopsy findings and anthropometric information used to estimate fall height through predictive ML models.

Previous studies showed promising results, but highlighted a key challenge: **model performance deteriorates as fall height increases**. The first study, based solely on the Milan dataset, discarded cases above 30 m due to limited data. The second study incorporated the Berlin dataset, which included more high-fall cases, and observed that while the model achieved a MAE of ~7 m on the full Berlin dataset, it performed significantly better when restricted to falls below 30 m.

This repository builds on those findings, aiming to further investigate these challenges and enhance the predictive power of ML techniques in this domain.

## Experiments

The analyses are structured in three phases:

1. **Height Cutoff Search** – Identifies an operational height cutoff beyond which predictive performance degrades significantly. The best-performing regression model (Kernel Ridge Regression) was evaluated across different candidate cutoffs, using both MAE and standard deviation to assess accuracy and stability.

2. **Height Classification** – Explores whether a ML model can reliably distinguish high from low fall heights. Multiple classification algorithms were assessed on the Berlin dataset alone and on a combined Berlin–Milan dataset, as an initial step toward a two-phase prediction system.

3. **Dual Evaluation** – Evaluates whether the Kernel Ridge Regression model can provide meaningful estimates below the operational cutoff while still offering actionable discrimination for high falls, by training on a dataset where heights are capped at the cutoff.


## Repository Structure

```
Fall-Height-Prediction-Using-ML/
├── height-cutoff-search/       # Regression pipeline: optimal height cutoff analysis
└── height-classification/      # Classification pipeline: binary height classification
```

## Projects Overview

### 1. Height Cutoff Search

An automated pipeline for analyzing the impact of different height cutoff values on fall height **regression** accuracy.

The pipeline runs in two steps:
1. **Data Generation** – generates datasets for each height cutoff configuration
2. **Model Experiments** – applies ML regression models to the generated datasets and evaluates performance using MAE

📖 See the full documentation in [`height-cutoff-search/README.md`](height-cutoff-search/README.md)

### 2. Height Classification

An automated pipeline for finding the best ML model for **binary classification** of fall height (above/below a given height threshold).

The pipeline applies multiple classification models to a single dataset and compares their performance using accuracy and F1-macro scores.

📖 See the full documentation in [`height-classification/README.md`](height-classification/README.md)

## Getting Started

### Prerequisites
- Python 3.8+

### Installation

Each project has its own dependencies. Install them separately:

```bash
# For height cutoff search
pip install -r height-cutoff-search/requirements.txt

# For height classification
pip install -r height-classification/requirements.txt
```

### Running the Pipelines

```bash
# Height cutoff search – full pipeline
cd height-cutoff-search
./run_experiments_for_height_cutoff.sh --both

# Height classification
cd height-classification
./run_experiments_for_height_classification.sh
