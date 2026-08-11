"""
Label frequency analysis for the Steam Review Aspect dataset.
Outputs:
  analysis/output/label_stats.csv   - per-label counts and frequencies for each split
  analysis/output/label_freq.png    - bar chart comparing train / test label frequencies
  analysis/output/label_cooccur.png - co-occurrence heatmap (train split)
"""

import json
import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

LABEL_NAMES = ["recommended", "story", "gameplay", "visual", "audio", "technical", "price", "suggestion"]
DATA_DIR = Path(__file__).parent.parent / "data" / "orig_data"
OUT_DIR = Path(__file__).parent / "output"

# ── Load data ──────────────────────────────────────────────────────────────────

def load_split(split: str) -> list[dict]:
    path = DATA_DIR / f"{split}.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]

train_data = load_split("train")
test_data  = load_split("test")

# ── Per-label counts and frequency ────────────────────────────────────────────

def label_counts(data: list[dict]) -> pd.DataFrame:
    n = len(data)
    counts = {label: sum(row["labels"][label] for row in data) for label in LABEL_NAMES}
    df = pd.DataFrame({
        "label":     list(counts.keys()),
        "count":     list(counts.values()),
        "frequency": [v / n for v in counts.values()],
    })
    df["total"] = n
    return df.set_index("label")

train_stats = label_counts(train_data)
test_stats  = label_counts(test_data)

combined = train_stats[["count", "frequency"]].rename(
    columns={"count": "train_count", "frequency": "train_freq"}
).join(test_stats[["count", "frequency"]].rename(
    columns={"count": "test_count", "frequency": "test_freq"}
))
combined.index.name = "label"
combined.to_csv(OUT_DIR / "label_stats.csv", float_format="%.4f")
print(combined.to_string())

# ── Per-split label count stats ───────────────────────────────────────────────

def labels_per_sample(data: list[dict]) -> list[int]:
    return [sum(row["labels"].values()) for row in data]

for split, data in [("train", train_data), ("test", test_data)]:
    counts = labels_per_sample(data)
    c = Counter(counts)
    print(f"\n{split} — labels per sample: "
          f"mean={np.mean(counts):.2f}, median={np.median(counts):.1f}, "
          f"min={min(counts)}, max={max(counts)}")
    print("  distribution:", dict(sorted(c.items())))

# ── Bar chart: train vs test label frequency ──────────────────────────────────

x = np.arange(len(LABEL_NAMES))
width = 0.38

fig, ax = plt.subplots(figsize=(11, 5))
bars_train = ax.bar(x - width / 2, combined["train_freq"], width, label="train (n=900)", color="#4C72B0")
bars_test  = ax.bar(x + width / 2, combined["test_freq"],  width, label="test  (n=200)", color="#DD8452")

ax.set_xticks(x)
ax.set_xticklabels(LABEL_NAMES, fontsize=11)
ax.set_ylabel("Frequency (proportion of samples)", fontsize=11)
ax.set_title("Label Frequency — Train vs Test", fontsize=13)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=10)
ax.yaxis.grid(True, linestyle="--", alpha=0.6)
ax.set_axisbelow(True)

for bar in [*bars_train, *bars_test]:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}",
            ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig(OUT_DIR / "label_freq.png", dpi=150)
plt.close()
print("\nSaved: label_freq.png")

# ── Co-occurrence heatmap (train) ─────────────────────────────────────────────

matrix = np.zeros((8, 8), dtype=int)
for row in train_data:
    vec = [row["labels"][l] for l in LABEL_NAMES]
    for i in range(8):
        for j in range(8):
            if vec[i] == 1 and vec[j] == 1:
                matrix[i][j] += 1

cooccur_df = pd.DataFrame(matrix, index=LABEL_NAMES, columns=LABEL_NAMES)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cooccur_df, annot=True, fmt="d", cmap="Blues",
    linewidths=0.5, ax=ax, cbar_kws={"label": "co-occurrence count"}
)
ax.set_title("Label Co-occurrence (train split)", fontsize=13)
plt.tight_layout()
plt.savefig(OUT_DIR / "label_cooccur.png", dpi=150)
plt.close()
print("Saved: label_cooccur.png")
