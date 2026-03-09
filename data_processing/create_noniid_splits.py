"""
Create Non-IID data splits for federated clients using Dirichlet distribution.
Phase 2: Realistic heterogeneous partitioning — no hardcoded class ratios.

The Dirichlet(alpha) approach:
  For each class c, sample proportions p ~ Dir(alpha * 1_K) and allocate
  that fraction of class-c samples to each client k.
  Lower alpha  → more heterogeneous (clients dominated by one class)
  Higher alpha → more uniform (closer to IID)
"""
import pickle
import numpy as np
import os

# ── Configuration ──────────────────────────────────────────────────────────────
CONFIG = {
    'n_clients':   10,
    'alpha':       0.5,    # Dirichlet concentration: lower = more heterogeneous
    'min_samples': 500,    # Hard floor: every client gets at least this many samples
    'random_seed': 42,
    'input_path':  'data/adult_train.pkl',
    'output_dir':  'data',
}

# ── Load training data ─────────────────────────────────────────────────────────
print("=" * 60)
print("CREATING NON-IID CLIENT DATA SPLITS (Dirichlet)")
print("=" * 60)

print(f"\n📥 Loading training data from {CONFIG['input_path']}...")
with open(CONFIG['input_path'], 'rb') as f:
    data = pickle.load(f)

X, y = data['X'], data['y']
feature_names = data['feature_names']
input_dim = data['input_dim']
print(f"✅ Loaded {len(X)} training samples, {input_dim} features")

n_clients = CONFIG['n_clients']
alpha = CONFIG['alpha']
classes = np.unique(y)

min_samples = CONFIG['min_samples']

print(f"\n⚙️  Configuration:")
print(f"   Clients:           {n_clients}")
print(f"   Dirichlet alpha:   {alpha}  (lower = more heterogeneous)")
print(f"   Min samples floor: {min_samples}")

# ── Dirichlet split ────────────────────────────────────────────────────────────
np.random.seed(CONFIG['random_seed'])

client_indices = [[] for _ in range(n_clients)]

for c in classes:
    class_idx = np.where(y == c)[0].copy()
    np.random.shuffle(class_idx)

    # Sample K proportions from Dirichlet; lower alpha = one client dominates
    proportions = np.random.dirichlet(alpha * np.ones(n_clients))

    # Convert to integer counts; fix rounding in last bucket
    counts = (proportions * len(class_idx)).astype(int)
    counts[-1] = len(class_idx) - counts[:-1].sum()
    counts = np.maximum(counts, 0)   # guard against negatives from rounding

    start = 0
    for k, count in enumerate(counts):
        client_indices[k].extend(class_idx[start:start + count].tolist())
        start += count

# ── Enforce minimum sample floor ──────────────────────────────────────────────
# Iteratively transfer samples from the largest client to any under-threshold
# client. This preserves the Dirichlet heterogeneity character while preventing
# pathologically tiny partitions that cause unstable local training.
def _enforce_min_samples(indices, floor, seed):
    rng = np.random.default_rng(seed)
    changed = True
    while changed:
        changed = False
        sizes = [len(c) for c in indices]
        for k in range(len(indices)):
            deficit = floor - sizes[k]
            if deficit <= 0:
                continue
            donor = int(np.argmax(sizes))
            if donor == k or sizes[donor] <= floor:
                continue   # donor is itself or already at floor
            arr = np.array(indices[donor])
            rng.shuffle(arr)
            indices[k].extend(arr[:deficit].tolist())
            indices[donor] = arr[deficit:].tolist()
            sizes[k] += deficit
            sizes[donor] -= deficit
            changed = True
    return indices

client_indices = _enforce_min_samples(client_indices, min_samples, CONFIG['random_seed'])

# Shuffle each client's local dataset
for k in range(n_clients):
    np.random.shuffle(client_indices[k])

# ── Distribution summary ───────────────────────────────────────────────────────
print(f"\n📊 Client Data Distribution (α={alpha}):")
print(f"{'Client':<10} {'Total':<10} {'Class 0':<12} {'Class 1':<12} {'Class 0 %':<12}")
print("-" * 60)

summary = []
for k, indices in enumerate(client_indices):
    idx = np.array(indices)
    client_y = y[idx]
    c0 = int((client_y == 0).sum())
    c1 = int((client_y == 1).sum())
    pct0 = c0 / len(client_y) * 100 if len(client_y) > 0 else 0.0
    summary.append({'client_id': k + 1, 'total': len(client_y), 'class0': c0, 'class1': c1, 'ratio0': pct0 / 100})
    print(f"Client {k+1:<3}  {len(idx):<10} {c0:<12} {c1:<12} {pct0:.1f}%")

print("-" * 60)

# ── Heterogeneity metrics ──────────────────────────────────────────────────────
ratios = [s['ratio0'] for s in summary]
totals = [s['total'] for s in summary]
print(f"\n📈 Heterogeneity Metrics:")
print(f"   Std Dev of Class-0 ratios: {np.std(ratios):.4f}  (higher = more Non-IID)")
print(f"   Min samples per client:    {min(totals)}")
print(f"   Max samples per client:    {max(totals)}")
print(f"   Total samples distributed: {sum(totals)} / {len(X)}")

# ── Save ───────────────────────────────────────────────────────────────────────
os.makedirs(CONFIG['output_dir'], exist_ok=True)
print(f"\n💾 Saving client data files...")

for k, indices in enumerate(client_indices):
    idx = np.array(indices)
    client_data = {
        'X': X[idx],
        'y': y[idx],
        'client_id': k + 1,
        'feature_names': feature_names,
        'input_dim': input_dim,
        'alpha': alpha,
    }
    path = os.path.join(CONFIG['output_dir'], f'client_{k + 1}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(client_data, f)
    print(f"   ✓ {path}  ({len(idx)} samples)")

# ── Phase 4 adversarial client recommendations ────────────────────────────────
# A good adversarial client for label-flipping needs:
#   (a) enough samples → high FedAvg weight → meaningful poisoning impact
#   (b) mixed class distribution → flipping labels actually changes predictions
# Score = n_samples * 2 * min(ratio0, ratio1)  [0 if fully one-class, max if balanced]
print(f"\n🎯 Phase 4: Adversarial Client Recommendations")
print(f"   (label-flipping impact = sample weight × class balance)")
print(f"{'Client':<10} {'Samples':<10} {'Class 0 %':<12} {'Impact Score':<14} {'Recommended':<12}")
print("-" * 60)
for s in sorted(summary, key=lambda x: -x['total'] * 2 * min(x['ratio0'], 1 - x['ratio0'])):
    impact = s['total'] * 2 * min(s['ratio0'], 1 - s['ratio0'])
    recommend = "✅ YES" if impact > 500 else "—"
    print(f"Client {s['client_id']:<3}  {s['total']:<10} {s['ratio0']*100:<12.1f} {impact:<14.0f} {recommend}")
print("-" * 60)
print("   Designate the top 2 'YES' clients as adversaries in Phase 4.")

print("\n" + "=" * 60)
print("✅ NON-IID CLIENT DATA CREATION COMPLETED!")
print("=" * 60)
