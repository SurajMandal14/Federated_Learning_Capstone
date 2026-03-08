# 📚 PROJECT FILES REFERENCE

## 📁 Complete File Structure

```
federated_privacy_project/
│
├── 📄 README.md                          # Project overview and quick start
├── 📄 DAY1_PROGRESS_REPORT.md           # Detailed progress report for meeting
├── 📄 DEMO_GUIDE.md                     # Step-by-step demo instructions
├── 📄 FILES_REFERENCE.md                # This file - explains all files
│
├── 📂 data/                             # Dataset and client data
│   ├── adult_processed.pkl              # Preprocessed UCI Adult dataset (1.4 MB)
│   ├── client_1.pkl                     # Client 1 data (2000 samples, 90:10 ratio)
│   ├── client_2.pkl                     # Client 2 data (2000 samples, 80:20 ratio)
│   ├── ...                              # Clients 3-9
│   └── client_10.pkl                    # Client 10 data (2000 samples, 50:50 ratio)
│
├── 📂 models/                           # Neural network architectures
│   ├── __init__.py                      # Package initializer
│   └── mlp_model.py                     # Simple MLP model (449 parameters)
│
├── 📂 data_processing/                  # Data preparation scripts
│   ├── download_data.py                 # Download & preprocess UCI Adult dataset
│   └── create_noniid_splits.py          # Create non-IID client data splits
│
├── 📂 experiments/                      # Experiment scripts
│   ├── baseline_fl_simple.py            # ⭐ MAIN DEMO - Working FL simulation
│   └── baseline_fl_demo.py              # Original version (has import issues)
│
├── 📂 results/                          # Output directory
│   ├── models/                          # Saved trained models
│   │   └── baseline_fl_model.pt         # Trained FL model (72.75% accuracy)
│   ├── plots/                           # Visualization outputs (empty for now)
│   └── logs/                            # Training logs (empty for now)
│
├── 📂 clients/                          # (Empty) Client implementation modules
├── 📂 server/                           # (Empty) Server-side logic
├── 📂 privacy/                          # (Empty) Differential privacy mechanisms
├── 📂 attacks/                          # (Empty) Adversarial attack simulations
├── 📂 evaluation/                       # (Empty) Metrics and evaluation tools
└── 📂 notebooks/                        # (Empty) Jupyter notebooks for analysis
```

---

## 🎯 WHAT EACH FILE DOES

### 📊 Documentation Files

#### README.md

- Project overview
- Quick start instructions
- Component status checklist
- Team information

#### DAY1_PROGRESS_REPORT.md ⭐

- Complete progress summary for meeting
- All results with numbers
- Next steps timeline
- Script for presentation

#### DEMO_GUIDE.md ⭐

- Step-by-step demo commands
- Expected outputs
- Presentation flow
- Q&A preparation

#### FILES_REFERENCE.md (This file)

- Complete file structure
- File descriptions
- Usage instructions

---

### 💾 Data Files

#### data/adult_processed.pkl

**What**: Preprocessed UCI Adult dataset
**Size**: 1.4 MB
**Contains**: 30,162 samples, 5 normalized features, binary labels
**Created by**: `data_processing/download_data.py`

#### data/client_1.pkl through client_10.pkl

**What**: Individual client data splits
**Size**: ~150 KB each
**Contains**: 2,000 samples per client with non-IID distribution
**Created by**: `data_processing/create_noniid_splits.py`

**Distribution**:

- Client 1: 90% Class 0, 10% Class 1 (heavy skew)
- Client 5: 50% Class 0, 50% Class 1 (balanced)
- Client 9: 10% Class 0, 90% Class 1 (heavy skew)

---

### 🧠 Model Files

#### models/mlp_model.py

**What**: Neural network architecture definition
**Type**: Simple MLP (Multi-Layer Perceptron)
**Architecture**: Input(5) → Linear(64) → ReLU → Linear(1) → Sigmoid
**Parameters**: 449 trainable parameters
**Purpose**: Binary classification for income prediction

**Key Class**: `SimpleMLPModel(nn.Module)`

- `forward()`: Forward pass
- `get_weights()`: Extract parameters
- `set_weights()`: Load parameters

---

### 🔧 Data Processing Scripts

#### data_processing/download_data.py

**Purpose**: Download and preprocess UCI Adult dataset
**Runtime**: ~30 seconds
**Output**: `data/adult_processed.pkl`

**Steps**:

1. Download from UCI repository
2. Remove missing values
3. Select numerical features
4. Normalize using StandardScaler
5. Encode binary target (>50K income)

**Usage**:

```bash
python data_processing/download_data.py
```

#### data_processing/create_noniid_splits.py

**Purpose**: Create non-IID client data distributions
**Runtime**: ~5 seconds
**Output**: 10 client pickle files

**Steps**:

1. Load preprocessed data
2. Separate by class
3. Create skewed distributions
4. Save per-client files

**Usage**:

```bash
python data_processing/create_noniid_splits.py
```

---

### 🔬 Experiment Scripts

#### experiments/baseline_fl_simple.py ⭐ MAIN DEMO

**Purpose**: Federated learning baseline simulation
**Runtime**: ~2 minutes
**Output**: Trained model + accuracy metrics

**Configuration**:

- 5 communication rounds
- 10 clients (all participate)
- 2 local epochs per round
- Learning rate: 0.01
- Batch size: 64

**Algorithms**:

- **FedAvg**: Federated averaging aggregation
- **SGD**: Local optimization
- **BCE Loss**: Binary cross-entropy

**Results**:

- Final accuracy: 72.75%
- Loss reduction: 0.5510 → 0.4918

**Usage**:

```bash
python experiments/baseline_fl_simple.py
```

**Key Functions**:

- `load_client_data()`: Load specific client data
- `train_client()`: Local training on client
- `federated_average()`: Aggregate client weights
- `evaluate_model()`: Calculate accuracy

---

### 💾 Saved Models

#### results/models/baseline_fl_model.pt

**What**: Trained federated learning model
**Size**: ~2 KB
**Format**: PyTorch state_dict
**Performance**: 72.75% average accuracy across 10 clients

**Load model**:

```python
import torch
from models.mlp_model import SimpleMLPModel

model = SimpleMLPModel(input_dim=5)
model.load_state_dict(torch.load('results/models/baseline_fl_model.pt'))
model.eval()
```

---

## 🚀 WHICH FILES TO RUN

### For Tomorrow's Demo:

1. **Show model works**:

   ```bash
   python models/mlp_model.py
   ```

   ⏱️ 10 seconds | ✅ Shows: Model created with 449 parameters

2. **Show data distribution**:

   ```bash
   python data_processing/create_noniid_splits.py
   ```

   ⏱️ 5 seconds | ✅ Shows: 10 clients with heterogeneous data

3. **Show FL training** (MAIN DEMO):
   ```bash
   python experiments/baseline_fl_simple.py
   ```
   ⏱️ 2 minutes | ✅ Shows: 5 rounds training, 72.75% accuracy

---

## 📝 WHICH FILES TO REFERENCE IN MEETING

### 1. DAY1_PROGRESS_REPORT.md

**Use for**: Detailed numbers and results

### 2. DEMO_GUIDE.md

**Use for**: Step-by-step presentation flow

### 3. README.md

**Use for**: Overview and next steps

---

## 🎯 FILE STATUSES

### ✅ Complete & Working

- [ ✓ ] README.md
- [ ✓ ] DAY1_PROGRESS_REPORT.md
- [ ✓ ] DEMO_GUIDE.md
- [ ✓ ] models/mlp_model.py
- [ ✓ ] data_processing/download_data.py
- [ ✓ ] data_processing/create_noniid_splits.py
- [ ✓ ] experiments/baseline_fl_simple.py
- [ ✓ ] data/adult_processed.pkl
- [ ✓ ] data/client\_\*.pkl (10 files)
- [ ✓ ] results/models/baseline_fl_model.pt

### 🔄 To Be Implemented (After GPU Access)

- [ ⏳ ] privacy/dp_mechanism.py (Differential Privacy)
- [ ⏳ ] attacks/label_flipping.py (Adversarial Clients)
- [ ⏳ ] server/aggregation.py (Robust Aggregation)
- [ ⏳ ] evaluation/metrics.py (Comprehensive Metrics)
- [ ⏳ ] interpretability/shap_analysis.py (SHAP)
- [ ⏳ ] experiments/exp_privacy.py (Privacy Experiment)
- [ ⏳ ] experiments/exp_attack.py (Attack Experiment)
- [ ⏳ ] experiments/exp_full_system.py (Full System)

---

## 🔍 IMPORTANT FILE CONTENTS

### Key Numbers to Remember:

| Metric                 | Value           |
| ---------------------- | --------------- |
| **Dataset samples**    | 30,162          |
| **Features**           | 5 (numerical)   |
| **Model parameters**   | 449             |
| **Clients**            | 10              |
| **Samples per client** | 2,000           |
| **Training rounds**    | 5               |
| **Final accuracy**     | 72.75%          |
| **Heterogeneity**      | 0.245 (Std Dev) |

---

## 💡 TIPS

### If Evaluator Asks "Can I see the code?"

→ Open: `experiments/baseline_fl_simple.py`
→ Point out: Clean structure, documented functions

### If Evaluator Asks "How do I run it?"

→ Show: `DEMO_GUIDE.md`
→ Say: "Just 3 commands"

### If Evaluator Asks "What's next?"

→ Show: `DAY1_PROGRESS_REPORT.md` → Next Steps section
→ Say: "Waiting on GPU for full experiments"

---

**Created**: March 8, 2026  
**Status**: All Day 1 deliverables complete ✅  
**Ready for**: Tomorrow's demo 🚀
