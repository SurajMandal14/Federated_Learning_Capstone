# Federated Learning: Privacy, Security, and Practical Challenges

## 🎯 Project Overview

This project implements a federated learning system with:

- **Privacy Protection**: Differential privacy mechanisms
- **Robustness**: Defense against adversarial attacks
- **Interpretability**: SHAP-based model explanations
- **Non-IID Data**: Realistic heterogeneous client distributions

## 📁 Project Structure

```
federated_privacy_project/
├── data/                      # Dataset and client data splits
├── models/                    # Neural network architectures
├── data_processing/           # Data preprocessing scripts
├── experiments/               # Experiment scripts
├── results/                   # Outputs (models, plots, logs)
└── README.md
```

## 🚀 Quick Start (Day 1 Demo)

### Step 1: Install Dependencies

```bash
pip install torch pandas scikit-learn matplotlib numpy
```

### Step 2: Download and Preprocess Data

```bash
cd federated_privacy_project
python data_processing/download_data.py
```

**Output**: Dataset downloaded and preprocessed (30,162 samples)

### Step 3: Test Model Architecture

```bash
python models/mlp_model.py
```

**Output**: Model created with 449 parameters

### Step 4: Create Non-IID Client Data

```bash
python data_processing/create_noniid_splits.py
```

**Output**: 10 clients created with heterogeneous distributions

### Step 5: Run Baseline Federated Learning

```bash
python experiments/baseline_fl_demo.py
```

**Output**: 5 rounds of federated training across 10 clients

## 📊 What This Demo Shows

✅ **Working Data Pipeline**: UCI Adult dataset preprocessed  
✅ **Model Architecture**: Lightweight MLP (449 parameters)  
✅ **Non-IID Setup**: 10 clients with varying class distributions  
✅ **Federated Training**: Simulated federated averaging (FedAvg)  
✅ **Ready for GPU**: Code structure ready for DGX deployment

## 🔜 Next Steps (After GPU Access)

1. **Differential Privacy**: Add noise to gradients (Opacus)
2. **Adversarial Clients**: Implement label flipping attacks
3. **Robust Aggregation**: Median/trimmed mean aggregation
4. **Full Experiments**: Run all 4 experiment configurations
5. **Interpretability**: SHAP analysis on trained model

## 📝 Research Components

| Component            | Status      | File                                      |
| -------------------- | ----------- | ----------------------------------------- |
| Baseline FL          | ✅ Complete | `experiments/baseline_fl_demo.py`         |
| Non-IID Data         | ✅ Complete | `data_processing/create_noniid_splits.py` |
| Differential Privacy | 🔄 Next     | `privacy/dp_mechanism.py`                 |
| Adversarial Attacks  | 🔄 Next     | `attacks/label_flipping.py`               |
| Robust Aggregation   | 🔄 Next     | `server/aggregation.py`                   |
| Interpretability     | 🔄 Next     | `interpretability/shap_analysis.py`       |

## 👥 Team

- Suraj Kumar Mandal Dhanuk (AP22110011480)
- Bibek Bhandari (AP22110011483)
- Shrijal Shrestha (AP22110011518)
- Rajveer Singh Khanduja (AP22110010166)

**Supervisor**: Dr. Saswat Kumar Ram

---

**Date**: March 8, 2026  
**Status**: Foundation complete, ready for DGX deployment
