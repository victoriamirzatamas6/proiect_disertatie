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

### Key Results (NASA CMAPSS FD001, window=30)

| Model | MAE (window) | RMSE (window) | MAE (unit-last) | RMSE (unit-last) |
|-------|-------------|--------------|----------------|-----------------|
| XGBoost | **10.54** | **14.04** | 10.22 | 13.24 |
| LSTM | **9.87** | 15.06 | 12.10 | 16.42 |

LSTM improves MAE by ~6% over the XGBoost baseline. XGBoost shows better
unit-last metrics, indicating stronger robustness in the critical near-failure zone.

**Anomaly Detection** (p99 threshold on test set):

| Method | Alarm Rate |
|--------|-----------|
| Autoencoder | 1.35% |
| PCA Baseline | 1.20% |

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

Place the following files inside `data/raw/`:

Required files:

-   train_FD001.txt
-   test_FD001.txt
-   RUL_FD001.txt

### 3.3 Processed Output

After running the pipeline, `data/processed/` will contain:

-   `train_normalized.csv` — Normalized training data
-   `valid_normalized.csv` — Normalized validation data
-   `test_normalized.csv` — Normalized test data
-   `xgb_features_w*.csv` — XGBoost features for window sizes (20, 30, 50)
-   `lstm_sequences_w*.csv` — LSTM sequence metadata for window sizes

### 3.4 Dataset Characteristics (FD001)

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

outputs/runs/runs.csv

This enables comparative benchmarking and reproducibility.

------------------------------------------------------------------------

## 7. Automated Reporting

After running the full pipeline, a report is generated automatically in
`outputs/report/`. You can also run it standalone:

```bash
python -m src.pipeline.make_report
```

Generated outputs:

-   `outputs/report/report.html`
-   `outputs/report/report.pdf`

The report includes: - Model performance comparison - Window-size
ablation results - Feature importance analysis - Anomaly threshold
evaluation - Alarm rate analysis - Experimental conclusions

------------------------------------------------------------------------

## 8. Installation

**Python 3.10+** required.

```bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 9. Running the Pipeline

```bash
python -m src.pipeline.run_all --config configs/config.yaml
```

The pipeline uses a simple cache by default and will skip rerunning work if the raw data and configuration are unchanged. Use `--force` to ignore cache and rerun everything.

Pipeline stages: 1. Data loading 2. Preprocessing 3. Window generation
4. Model training 5. Evaluation 6. Explainability 7. Anomaly detection
8. Artifact storage 9. Report generation

------------------------------------------------------------------------

## 10. Interactive Dashboard

```bash
streamlit run app/streamlit_app.py
```

Dashboard capabilities: - Model selection (XGB / LSTM) - Window-size
comparison - Threshold selection (95/97/99) - Visualization of RUL
predictions - Feature importance plots - Reconstruction error analysis -
Export of results

------------------------------------------------------------------------

## 11. Project Structure

```
pdm-platform-plus/
│
├── app/
│   └── streamlit_app.py          # Interactive Streamlit dashboard
│
├── configs/
│   └── config.yaml               # Centralized experiment configuration
│
├── data/
│   ├── raw/
│   │   ├── train_FD001.txt
│   │   ├── test_FD001.txt
│   │   └── RUL_FD001.txt
│   └── processed/                # Processed data output
│       ├── train_normalized.csv
│       ├── valid_normalized.csv
│       ├── test_normalized.csv
│       ├── xgb_features_w*.csv
│       └── lstm_sequences_w*.csv
│
├── docs/
│   └── Documentatie_*.docx       # Dissertation document (gitignored)
│
├── outputs/                      # All generated artifacts (gitignored)
│   ├── anomaly/
│   ├── figures/
│   ├── metrics/
│   ├── models/
│   ├── predictions/
│   ├── report/
│   └── runs/
│
├── scripts/                      # Standalone figure generation scripts
│   ├── compute_metrics.py
│   ├── fig_6_2_correlation.py
│   ├── fig_6_3_sensor_trends.py
│   ├── fig_6_4_rul_distribution.py
│   ├── fig_6_5_xgb_scatter.py
│   ├── fig_6_11_error_distribution.py
│   ├── fig_6_12_error_vs_rul.py
│   └── make_fig_6_13.py
│
├── src/
│   ├── data/
│   │   ├── cmapss_loader.py      # Dataset loading
│   │   ├── features.py           # Window feature engineering
│   │   ├── preprocess.py         # Normalization / scaling
│   │   └── split.py              # Train/validation split by unit
│   ├── experiments/
│   │   ├── report.py             # HTML/PDF report generation
│   │   └── tracking.py           # Experiment versioning
│   ├── models/
│   │   ├── anomaly_autoencoder.py
│   │   ├── anomaly_pca.py
│   │   ├── baseline_xgb.py
│   │   ├── explain_xgb.py
│   │   └── lstm_rul.py
│   ├── pipeline/
│   │   ├── make_report.py        # Standalone report generation entry point
│   │   └── run_all.py            # Full pipeline entry point
│   └── utils/
│       ├── io.py
│       ├── metrics.py
│       ├── plotting.py
│       └── seed.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

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
