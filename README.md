# PdM Platform+

## A Reproducible Predictive Maintenance Framework on NASA CMAPSS (FD001)

------------------------------------------------------------------------

## 1. Abstract

**PdM Platform+** is a modular and reproducible predictive maintenance
framework developed for Remaining Useful Life (RUL) estimation and
anomaly detection using the NASA CMAPSS FD001 turbofan engine dataset.

The platform integrates:

-   Gradient Boosted Decision Trees (XGBoost)
-   Long Short-Term Memory (LSTM) neural networks
-   Window-size ablation studies (20 / 30 / 50 cycles)
-   Multi-level evaluation metrics (window-level and unit-level)
-   Model explainability (gain-based importance + permutation
    importance)
-   Autoencoder-based anomaly detection
-   PCA reconstruction baseline comparison
-   Threshold-based alarm rate analysis (95/97/99 percentiles)
-   Automated experiment tracking
-   Automated PDF/HTML reporting
-   Interactive Streamlit dashboard

The framework is designed for research reproducibility, industrial-grade
experimentation, and explainable AI in predictive maintenance.

------------------------------------------------------------------------

## 2. Problem Formulation

### 2.1 Remaining Useful Life (RUL) Estimation

Given multivariate time-series sensor data from turbofan engines
operating under degradation conditions, the objective is to predict:

RUL(t) = T_failure − t

Where: - T_failure = last cycle before failure - t = current time step

The problem is treated as a supervised regression task.

### 2.2 Anomaly Detection

Anomaly detection is framed as: - Reconstruction-error-based detection -
Unsupervised learning on normal training distribution - Threshold-based
alarm generation

Two approaches are compared: 1. Deep Autoencoder 2. PCA Reconstruction
Baseline

------------------------------------------------------------------------

## 3. Dataset

### 3.1 Source

NASA CMAPSS -- FD001 subset\
Simulated turbofan engine degradation dataset.

### 3.2 Required Files

Place the following files inside:

data/raw/

Required files:

-   train_FD001.txt
-   test_FD001.txt
-   RUL_FD001.txt

### 3.3 Dataset Characteristics (FD001)

-   100 training engines
-   100 test engines
-   21 sensor measurements
-   3 operational settings
-   Single degradation mode
-   Single operating condition

------------------------------------------------------------------------

## 4. Methodology

### 4.1 Feature Engineering

-   Sliding window segmentation
-   Window sizes evaluated: 20, 30, 50
-   Sensor normalization
-   Optional RUL capping

### 4.2 RUL Prediction Models

#### 4.2.1 XGBoost Regressor

-   Tree-based ensemble method
-   Gradient boosting optimization
-   Robust to non-linear degradation patterns

Explainability: - Gain-based feature importance - Permutation importance

#### 4.2.2 LSTM Network

-   Sequential deep learning model
-   Captures temporal degradation dependencies
-   Trained on windowed sequences
-   Optimized using MSE loss

### 4.3 Evaluation Strategy

Two evaluation granularities:

#### Window-Level Metrics

-   RMSE
-   MAE
-   R²

#### Unit-Level Metrics (Last Cycle Only)

Prediction evaluated only at final cycle of each test engine.

Metrics: - RMSE - MAE

This reflects realistic industrial deployment scenarios.

------------------------------------------------------------------------

## 5. Anomaly Detection

### 5.1 Autoencoder

-   Trained on normal training data
-   Reconstruction error used as anomaly score

Thresholds evaluated: - 95th percentile - 97th percentile - 99th
percentile

Outputs: - Alarm rate - Reconstruction distributions

### 5.2 PCA Baseline

-   Dimensionality reduction
-   Reconstruction-based anomaly score
-   Baseline comparison with deep model

------------------------------------------------------------------------

## 6. Experiment Tracking

All experiments are versioned automatically:

outputs/runs/`<timestamp>`{=html}/

Tracked artifacts: - Model parameters - Evaluation metrics - Feature
importance plots - Anomaly distributions - Configuration snapshot

Global summary file:

outputs/runs.csv

This enables comparative benchmarking and reproducibility.

------------------------------------------------------------------------

## 7. Automated Reporting

After running the full pipeline:

python -m src.pipeline.make_report --pdf

Generated outputs:

-   outputs/report/report.html
-   outputs/report/report.pdf (if reportlab installed)

The report includes: - Model performance comparison - Window-size
ablation results - Feature importance analysis - Anomaly threshold
evaluation - Alarm rate analysis - Experimental conclusions

------------------------------------------------------------------------

## 8. Installation

Python 3.9+

pip install -r requirements.txt

------------------------------------------------------------------------

## 9. Running the Pipeline

python -m src.pipeline.run_all --config configs/config.yaml

Pipeline stages: 1. Data loading 2. Preprocessing 3. Window generation
4. Model training 5. Evaluation 6. Explainability 7. Anomaly detection
8. Artifact storage 9. Report generation

------------------------------------------------------------------------

## 10. Interactive Dashboard

streamlit run app/streamlit_app.py

Dashboard capabilities: - Model selection (XGB / LSTM) - Window-size
comparison - Threshold selection (95/97/99) - Visualization of RUL
predictions - Feature importance plots - Reconstruction error analysis -
Export of results

------------------------------------------------------------------------

## 11. Project Structure

pdm-platform-plus/ │ ├── data/ │ └── raw/ ├── configs/ │ └── config.yaml
├── src/ ├── app/ ├── outputs/ ├── requirements.txt └── README.md

------------------------------------------------------------------------

## 12. Reproducibility

The framework ensures: - Deterministic seeds - Config-driven
experiments - Versioned artifacts - Automated reporting

------------------------------------------------------------------------

## 13. Future Extensions

-   Multi-dataset support (FD002--FD004)
-   Attention-based sequence models
-   Bayesian uncertainty estimation
-   Online inference API
-   MLOps integration (MLflow / Docker)

------------------------------------------------------------------------

## 14. Citation

If used in academic or industrial work, please cite:

PdM Platform+: A Reproducible Predictive Maintenance Framework on NASA
CMAPSS FD001, 2026.
