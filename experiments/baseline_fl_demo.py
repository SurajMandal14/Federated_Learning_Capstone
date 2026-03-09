"""
Baseline Federated Learning Simulation (No Privacy, No Attacks).
Phase 2: Proper FL with held-out test evaluation, per-round CSV logging,
         weighted FedAvg, and convergence plots.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import pickle
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

from models.mlp_model import MLPModel

try:
    import matplotlib
    matplotlib.use('Agg')   # non-interactive backend (safe for all environments)
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from sklearn.metrics import f1_score, precision_score, recall_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)

# ── Configuration (all tuneable here, nothing hardcoded below) ─────────────────
CONFIG = {
    'n_rounds':          10,
    'n_clients':         10,
    'clients_per_round': 10,   # set < n_clients for partial participation
    'local_epochs':       2,
    'local_batch_size':  64,
    'learning_rate':     0.01,
    'hidden_dims':       [128, 64],
    'dropout_rate':      0.2,
    'random_seed':       42,
    'data_dir':          _path('data'),
    'results_dir':       _path('results'),
}

torch.manual_seed(CONFIG['random_seed'])
np.random.seed(CONFIG['random_seed'])

# ── Data helpers ───────────────────────────────────────────────────────────────
def load_client_data(client_id: int):
    path = os.path.join(CONFIG['data_dir'], f'client_{client_id}.pkl')
    with open(path, 'rb') as f:
        d = pickle.load(f)
    return d['X'], d['y']


def load_test_data():
    path = os.path.join(CONFIG['data_dir'], 'adult_test.pkl')
    with open(path, 'rb') as f:
        d = pickle.load(f)
    return d['X'], d['y'], d['input_dim']

# ── Training ───────────────────────────────────────────────────────────────────
def train_client(model, X, y, epochs, lr, batch_size):
    model.train()
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    X_t = torch.FloatTensor(X)
    y_t = torch.FloatTensor(y).unsqueeze(1)
    n = len(X_t)
    final_loss = 0.0

    for _ in range(epochs):
        perm = torch.randperm(n)
        X_t, y_t = X_t[perm], y_t[perm]
        epoch_loss, n_batches = 0.0, 0
        for start in range(0, n, batch_size):
            bX = X_t[start:start + batch_size]
            by = y_t[start:start + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(bX), by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        final_loss = epoch_loss / max(n_batches, 1)

    return model.state_dict(), final_loss

# ── Evaluation ─────────────────────────────────────────────────────────────────
def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X)
        y_t = torch.FloatTensor(y).unsqueeze(1)
        probs = model(X_t)
        preds = (probs > 0.5).float()
        accuracy = (preds == y_t).float().mean().item()
        loss = nn.BCELoss()(probs, y_t).item()

    metrics = {'accuracy': accuracy, 'loss': loss}

    if HAS_SKLEARN:
        preds_np = preds.numpy().flatten().astype(int)
        metrics['f1']        = f1_score(y, preds_np, zero_division=0)
        metrics['precision'] = precision_score(y, preds_np, zero_division=0)
        metrics['recall']    = recall_score(y, preds_np, zero_division=0)

    return metrics

# ── Weighted FedAvg ────────────────────────────────────────────────────────────
def federated_average(client_weights, client_sizes):
    """Weighted average: each client contributes proportional to its sample count."""
    total = sum(client_sizes)
    avg = {}
    for key in client_weights[0].keys():
        avg[key] = torch.zeros_like(client_weights[0][key].float())
        for w, n in zip(client_weights, client_sizes):
            avg[key] += w[key].float() * (n / total)
    return avg

# ── CSV logging ────────────────────────────────────────────────────────────────
def init_csv(path, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def append_csv(path, row):
    with open(path, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=list(row.keys())).writerow(row)

# ── Convergence plot ───────────────────────────────────────────────────────────
def plot_convergence(csv_path, plot_dir):
    if not HAS_MATPLOTLIB:
        print("   ⚠️  matplotlib not available — skipping plots")
        return

    rounds, accs, losses, f1s = [], [], [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rounds.append(int(row['round']))
            accs.append(float(row['test_accuracy']))
            losses.append(float(row['avg_train_loss']))
            if 'test_f1' in row:
                f1s.append(float(row['test_f1']))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(rounds, accs, 'b-o', label='Test Accuracy')
    if f1s:
        ax1.plot(rounds, f1s, 'g-s', label='Test F1')
    ax1.set(xlabel='Communication Round', ylabel='Score',
            title='Baseline FL — Accuracy & F1 per Round', ylim=(0, 1))
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(rounds, losses, 'r-o', label='Avg Train Loss')
    ax2.set(xlabel='Communication Round', ylabel='BCE Loss',
            title='Baseline FL — Training Loss per Round')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, 'baseline_convergence.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"   📊 Saved: {plot_path}")

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print(" " * 15 + "FEDERATED LEARNING BASELINE SIMULATION")
    print("=" * 70)

    # Load held-out test set and derive input_dim from data (no hardcoding)
    X_test, y_test, input_dim = load_test_data()
    print(f"\n📂 Held-out test set: {len(X_test)} samples, {input_dim} features")

    print(f"\n⚙️  Configuration:")
    for k, v in CONFIG.items():
        print(f"   {k}: {v}")

    # Build global model
    global_model = MLPModel(
        input_dim=input_dim,
        hidden_dims=CONFIG['hidden_dims'],
        dropout_rate=CONFIG['dropout_rate'],
    )
    total_params = sum(p.numel() for p in global_model.parameters())
    arch_str = f"{input_dim} → {' → '.join(map(str, CONFIG['hidden_dims']))} → 1"
    print(f"\n🌍 Global model: {total_params} parameters  [{arch_str}]")

    # Set up logging paths
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir   = os.path.join(CONFIG['results_dir'], 'logs')
    plot_dir  = os.path.join(CONFIG['results_dir'], 'plots')
    model_dir = os.path.join(CONFIG['results_dir'], 'models')
    csv_path  = os.path.join(log_dir, f'baseline_metrics_{timestamp}.csv')

    fieldnames = ['round', 'avg_train_loss', 'test_loss', 'test_accuracy']
    if HAS_SKLEARN:
        fieldnames += ['test_f1', 'test_precision', 'test_recall']
    init_csv(csv_path, fieldnames)
    print(f"\n📝 Logging metrics to: {csv_path}")

    # RNG for client selection (reproducible)
    rng = np.random.default_rng(CONFIG['random_seed'])

    print("\n" + "=" * 70)
    print("TRAINING START")
    print("=" * 70)

    for round_num in range(1, CONFIG['n_rounds'] + 1):
        print(f"\n📍 Round {round_num}/{CONFIG['n_rounds']}")
        print("-" * 70)

        # Random client subset each round
        selected = rng.choice(
            CONFIG['n_clients'],
            size=CONFIG['clients_per_round'],
            replace=False,
        ) + 1   # client IDs are 1-indexed

        client_weights, client_sizes, client_losses = [], [], []

        for cid in sorted(selected):
            X_c, y_c = load_client_data(int(cid))

            local_model = MLPModel(
                input_dim=input_dim,
                hidden_dims=CONFIG['hidden_dims'],
                dropout_rate=CONFIG['dropout_rate'],
            )
            local_model.load_state_dict(global_model.state_dict())

            weights, loss = train_client(
                local_model, X_c, y_c,
                epochs=CONFIG['local_epochs'],
                lr=CONFIG['learning_rate'],
                batch_size=CONFIG['local_batch_size'],
            )
            client_weights.append(weights)
            client_sizes.append(len(X_c))
            client_losses.append(loss)
            print(f"   Client {cid:2d} | Loss: {loss:.4f} | Samples: {len(X_c)}")

        # Weighted FedAvg
        global_model.load_state_dict(federated_average(client_weights, client_sizes))
        avg_train_loss = float(np.mean(client_losses))

        # Evaluate on held-out test set every round
        metrics = evaluate(global_model, X_test, y_test)

        print(f"\n   🔄 Aggregated {len(client_weights)} client updates (weighted FedAvg)")
        print(f"   📉 Avg Train Loss:  {avg_train_loss:.4f}")
        print(f"   🎯 Test Accuracy:   {metrics['accuracy']*100:.2f}%", end='')
        if 'f1' in metrics:
            print(f"   |  F1: {metrics['f1']:.4f}   "
                  f"Precision: {metrics['precision']:.4f}   "
                  f"Recall: {metrics['recall']:.4f}", end='')
        print()

        row = {
            'round':           round_num,
            'avg_train_loss':  round(avg_train_loss,        6),
            'test_loss':       round(metrics['loss'],       6),
            'test_accuracy':   round(metrics['accuracy'],   6),
        }
        if HAS_SKLEARN:
            row['test_f1']        = round(metrics.get('f1',        0), 6)
            row['test_precision'] = round(metrics.get('precision', 0), 6)
            row['test_recall']    = round(metrics.get('recall',    0), 6)
        append_csv(csv_path, row)

    # Save final model with full config
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'baseline_fl_{timestamp}.pt')
    torch.save({
        'model_state_dict': global_model.state_dict(),
        'config':           CONFIG,
        'input_dim':        input_dim,
        'timestamp':        timestamp,
    }, model_path)

    # Generate convergence plots
    print(f"\n📈 Generating convergence plots...")
    plot_convergence(csv_path, plot_dir)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    print(f"\n💾 Model:   {model_path}")
    print(f"📝 Metrics: {csv_path}")

    print("\n" + "=" * 70)
    print("✅ FEDERATED LEARNING SIMULATION SUCCESSFUL!")
    print("=" * 70)
    print("\n📝 Next Steps:")
    print("   1. Add differential privacy     → privacy/dp_mechanism.py")
    print("   2. Simulate adversarial clients → attacks/label_flipping.py")
    print("   3. Add robust aggregation       → server/aggregation.py")
    print("   4. Run SHAP explainability      → analysis/shap_analysis.py")
    print("=" * 70)
