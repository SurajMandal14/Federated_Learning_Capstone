"""
Download and preprocess UCI Adult dataset for federated learning.
Phase 1: Full feature engineering with stratified train/test split.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os

# ── Configuration ──────────────────────────────────────────────────────────────
CONFIG = {
    'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
    'test_size': 0.20,
    'random_seed': 42,
    'output_dir': 'data',
}

COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income',
]

# fnlwgt: census weight (not predictive); education: redundant with education_num
DROP_COLS = ['fnlwgt', 'education']

NUMERICAL_COLS = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']

# One-hot encoded (drop_first=True avoids multicollinearity)
CATEGORICAL_COLS = ['workclass', 'marital_status', 'occupation', 'relationship', 'race']

# ── Download ───────────────────────────────────────────────────────────────────
print("=" * 60)
print("DOWNLOADING AND PREPROCESSING UCI ADULT DATASET")
print("=" * 60)

print(f"\nDownloading dataset from UCI repository...")
df = pd.read_csv(
    CONFIG['url'],
    names=COLUMNS,
    sep=r',\s*',
    engine='python',
    na_values='?',
)
print(f"Downloaded {len(df)} samples")

# ── Clean ──────────────────────────────────────────────────────────────────────
print("\n🧹 Cleaning data (removing missing values, dropping redundant columns)...")
df = df.dropna()
df = df.drop(columns=DROP_COLS)
print(f"After cleaning: {len(df)} samples, {len(df.columns) - 1} raw features")

# ── Encode target ──────────────────────────────────────────────────────────────
y = df['income'].apply(lambda x: 1 if '>50K' in x else 0).values
df = df.drop(columns=['income'])

# ── Binary-encode sex and native_country ──────────────────────────────────────
df['sex'] = df['sex'].map({'Male': 1, 'Female': 0}).astype(float)
df['native_country'] = df['native_country'].apply(
    lambda x: 1.0 if x == 'United-States' else 0.0
)

# ── One-hot encode categorical columns ────────────────────────────────────────
df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True, dtype=float)

feature_names = list(df.columns)
X = df.values.astype(float)
input_dim = len(feature_names)

print(f"\nFeature Engineering:")
print(f"   Numerical (scaled): {NUMERICAL_COLS}")
print(f"   Binary encoded:     sex, native_country")
print(f"   One-hot encoded:    {CATEGORICAL_COLS}")
print(f"   Total features after encoding: {input_dim}")

# ── Stratified train / test split ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=CONFIG['test_size'],
    stratify=y,
    random_state=CONFIG['random_seed'],
)
print(f"\nStratified Train/Test Split ({int((1-CONFIG['test_size'])*100)}/{int(CONFIG['test_size']*100)}):")
print(f"   Train: {len(X_train)} samples")
print(f"   Test:  {len(X_test)} samples  ← held-out, never seen by clients")

# ── Normalize (fit on train only, apply to both) ───────────────────────────────
print("\n🔧 Normalizing features (StandardScaler fit on train set only)...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ── Class distribution ─────────────────────────────────────────────────────────
for split_name, y_split in [('Train', y_train), ('Test', y_test)]:
    c0 = (y_split == 0).sum()
    c1 = (y_split == 1).sum()
    print(f"\n{split_name} Class Distribution:")
    print(f"   <=50K (Class 0): {c0} samples ({c0 / len(y_split) * 100:.1f}%)")
    print(f"   >50K  (Class 1): {c1} samples ({c1 / len(y_split) * 100:.1f}%)")

# ── Save ───────────────────────────────────────────────────────────────────────
os.makedirs(CONFIG['output_dir'], exist_ok=True)

metadata = {
    'feature_names': feature_names,
    'input_dim': input_dim,
    'numerical_cols': NUMERICAL_COLS,
    'categorical_cols': CATEGORICAL_COLS,
    'n_classes': 2,
    'random_seed': CONFIG['random_seed'],
}

with open(f"{CONFIG['output_dir']}/adult_train.pkl", 'wb') as f:
    pickle.dump({'X': X_train, 'y': y_train, 'scaler': scaler, **metadata}, f)

with open(f"{CONFIG['output_dir']}/adult_test.pkl", 'wb') as f:
    pickle.dump({'X': X_test, 'y': y_test, **metadata}, f)

print(f"\nSaved:")
print(f"   {CONFIG['output_dir']}/adult_train.pkl  ({len(X_train)} samples, {input_dim} features)")
print(f"   {CONFIG['output_dir']}/adult_test.pkl   ({len(X_test)} samples, {input_dim} features)")

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED SUCCESSFULLY!")
print("=" * 60)
