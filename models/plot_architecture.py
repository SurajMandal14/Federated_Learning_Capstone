"""
Generate a visual flowchart of the MLPModel architecture.
Saves to results/plots/model_architecture.png
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Layer definitions ──────────────────────────────────────────────────────────
LAYERS = [
    {"label": "INPUT",          "sublabel": "41 features",          "params": "",           "color": "#4A90D9", "text": "white"},
    {"label": "Linear",         "sublabel": "41 → 128",             "params": "5,376 params","color": "#2ECC71", "text": "white"},
    {"label": "ReLU",           "sublabel": "Activation",           "params": "",           "color": "#F39C12", "text": "white"},
    {"label": "Dropout",        "sublabel": "p = 0.2",              "params": "",           "color": "#E74C3C", "text": "white"},
    {"label": "Linear",         "sublabel": "128 → 64",             "params": "8,256 params","color": "#2ECC71", "text": "white"},
    {"label": "ReLU",           "sublabel": "Activation",           "params": "",           "color": "#F39C12", "text": "white"},
    {"label": "Dropout",        "sublabel": "p = 0.2",              "params": "",           "color": "#E74C3C", "text": "white"},
    {"label": "Linear",         "sublabel": "64 → 1",               "params": "65 params",  "color": "#2ECC71", "text": "white"},
    {"label": "Sigmoid",        "sublabel": "Activation",           "params": "",           "color": "#F39C12", "text": "white"},
    {"label": "OUTPUT",         "sublabel": "P(income > 50K)",      "params": "",           "color": "#9B59B6", "text": "white"},
]

# ── Canvas setup ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 14))
ax.set_xlim(0, 6)
ax.set_ylim(0, len(LAYERS) * 1.2 + 0.5)
ax.axis('off')
fig.patch.set_facecolor('#1A1A2E')
ax.set_facecolor('#1A1A2E')

BOX_W  = 4.0
BOX_H  = 0.75
X_CTR  = 3.0
STRIDE = 1.2

def draw_box(ax, cx, cy, layer):
    x = cx - BOX_W / 2
    y = cy - BOX_H / 2
    box = FancyBboxPatch(
        (x, y), BOX_W, BOX_H,
        boxstyle="round,pad=0.08",
        linewidth=1.5,
        edgecolor="white",
        facecolor=layer["color"],
        zorder=3,
    )
    ax.add_patch(box)

    # Main label (bold)
    ax.text(cx, cy + 0.12, layer["label"],
            ha='center', va='center',
            fontsize=11, fontweight='bold',
            color=layer["text"], zorder=4)

    # Sublabel
    ax.text(cx, cy - 0.14, layer["sublabel"],
            ha='center', va='center',
            fontsize=8.5, color=layer["text"],
            alpha=0.9, zorder=4)

    # Param count (right-aligned inside box)
    if layer["params"]:
        ax.text(x + BOX_W - 0.15, cy, layer["params"],
                ha='right', va='center',
                fontsize=7.5, color=layer["text"],
                alpha=0.75, style='italic', zorder=4)

# ── Draw layers bottom-to-top (index 0 = top visually) ────────────────────────
n = len(LAYERS)
for i, layer in enumerate(LAYERS):
    cy = (n - i) * STRIDE
    draw_box(ax, X_CTR, cy, layer)

    # Arrow to next box
    if i < n - 1:
        arrow_start_y = cy - BOX_H / 2
        arrow_end_y   = (n - i - 1) * STRIDE + BOX_H / 2
        ax.annotate(
            "", xy=(X_CTR, arrow_end_y), xytext=(X_CTR, arrow_start_y),
            arrowprops=dict(arrowstyle="-|>", color="white",
                            lw=1.5, mutation_scale=14),
            zorder=2,
        )

# ── Output shape labels on the right ──────────────────────────────────────────
SHAPES = [
    "(batch, 41)", "(batch, 128)", "(batch, 128)", "(batch, 128)",
    "(batch, 64)",  "(batch, 64)",  "(batch, 64)",
    "(batch, 1)",   "(batch, 1)",   "(batch, 1)",
]
for i, shape in enumerate(SHAPES):
    cy = (n - i) * STRIDE
    ax.text(X_CTR + BOX_W / 2 + 0.15, cy, shape,
            ha='left', va='center', fontsize=7.5,
            color='#AAAAAA', zorder=4)

# ── Legend ─────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(color='#4A90D9', label='I/O'),
    mpatches.Patch(color='#2ECC71', label='Linear (learnable)'),
    mpatches.Patch(color='#F39C12', label='Activation'),
    mpatches.Patch(color='#E74C3C', label='Dropout (regularisation)'),
    mpatches.Patch(color='#9B59B6', label='Output'),
]
ax.legend(handles=legend_items, loc='lower center',
          bbox_to_anchor=(0.5, -0.01),
          ncol=2, fontsize=8,
          facecolor='#2C2C4E', edgecolor='white',
          labelcolor='white', framealpha=0.9)

# ── Title & footer ─────────────────────────────────────────────────────────────
ax.set_title('MLPModel Architecture\n41 → 128 → 64 → 1',
             fontsize=13, fontweight='bold',
             color='white', pad=12)

ax.text(X_CTR, 0.25,
        'Total params: 13,697  |  Size: 0.05 MB  |  All params trainable',
        ha='center', va='center', fontsize=8,
        color='#AAAAAA', style='italic')

# ── Save ───────────────────────────────────────────────────────────────────────
out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'results', 'plots')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'model_architecture.png')
plt.tight_layout()
plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"✅ Saved: {out_path}")
