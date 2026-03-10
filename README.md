# Federated Learning: Privacy, Security, and Practical Challenges

## Project Overview

A federated learning framework built on the UCI Adult Census dataset that investigates the interplay between **privacy**, **robustness**, **explainability**, and **non-IID data heterogeneity** — areas typically studied in isolation.

### Research Goal

Combine differential privacy (DP-SGD), Byzantine-robust aggregation, and SHAP explainability within a single federated system operating under realistic non-IID conditions, and measure how each mechanism affects the others.

## Current Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Data pipeline & feature engineering (41 features, stratified split) | Complete |
| 2 | Baseline FedAvg with non-IID splits & per-round evaluation | Complete |
| 3 | Differential privacy (DP-SGD with Opacus, ε ∈ {0.1, 1.0, 10.0}) | Planned |
| 4 | Adversarial attacks & robust aggregation (median, trimmed mean) | Planned |
| 5 | SHAP explainability across all configurations | Planned |
| 6 | Results analysis & paper write-up | Planned |

**Baseline accuracy: 84.15%** (10 rounds, weighted FedAvg, Non-IID Dirichlet α=0.5)

## Project Structure

```
Federated_Learning_Capstone/
├── data_processing/
│   ├── download_data.py            # UCI Adult download, 41-feature encoding, train/test split
│   └── create_noniid_splits.py     # Dirichlet(α=0.5) non-IID partitioning, min-samples floor
├── models/
│   ├── mlp_model.py                # MLP architecture (41→128→64→1), model summary
│   └── plot_architecture.py        # Visual model architecture flowchart
├── experiments/
│   └── baseline_fl_demo.py         # FedAvg training loop with per-round evaluation & logging
├── results/
│   ├── logs/                       # Per-round CSV metrics (loss, accuracy, F1, precision, recall)
│   ├── models/                     # Saved model checkpoints (.pt)
│   └── plots/                      # Convergence plots, architecture diagrams
├── .gitignore
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
pip install torch pandas scikit-learn matplotlib numpy
```

### 2. Download and preprocess data

```bash
python data_processing/download_data.py
```

Downloads the UCI Adult dataset, applies full feature engineering (one-hot + binary encoding, 41 features), performs a stratified 80/20 train/test split, fits the scaler on training data only, and saves `adult_train.pkl` and `adult_test.pkl`.

### 3. Create non-IID client splits

```bash
python data_processing/create_noniid_splits.py
```

Partitions training data across 10 clients using Dirichlet(α=0.5) distribution. Enforces a minimum of 500 samples per client via a donor-transfer algorithm. Computes adversarial impact scores for Phase 4 client selection.

### 4. Run baseline federated learning

```bash
python experiments/baseline_fl_demo.py
```

Runs 10 rounds of weighted FedAvg across all 10 clients. Evaluates on a held-out test set every round. Logs metrics to CSV and generates convergence plots.

## Technical Details

### Data Pipeline
- **Dataset**: UCI Adult Census (32,561 samples, binary classification: income >$50K)
- **Features**: 41 after encoding (dropped `fnlwgt`, `education`; one-hot for categoricals; binary for sex, native_country)
- **Split**: Stratified 80/20 train/test; scaler fit on train only

### Model Architecture
- **Type**: Multi-Layer Perceptron (MLP)
- **Layers**: Input(41) → Linear(128) → ReLU → Dropout(0.2) → Linear(64) → ReLU → Dropout(0.2) → Linear(1) → Sigmoid
- **Parameters**: ~11,201
- **Loss**: Binary Cross-Entropy
- **Optimizer**: SGD (lr=0.01)

### Federated Setup
- **Algorithm**: FedAvg with weighted aggregation (weights ∝ client sample count)
- **Clients**: 10, full participation each round
- **Non-IID**: Dirichlet(α=0.5) label distribution with 500-sample floor
- **Local training**: 2 epochs, batch size 64 per round
- **Evaluation**: Held-out test set every round (accuracy, F1, precision, recall)

## Team

- Suraj Kumar Mandal Dhanuk (AP22110011480)
- Bibek Bhandari (AP22110011483)
- Shrijal Shrestha (AP22110011518)
- Rajveer Singh Khanduja (AP22110010166)

**Supervisor**: Dr. Saswat Kumar Ram
