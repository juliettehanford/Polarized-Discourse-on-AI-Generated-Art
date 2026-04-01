"""
Generate t-SNE visualization of user SBERT embeddings colored by stance label.

Outputs:
  results/tsne_embeddings.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    features = torch.load(DATA_DIR / "processed" / "node_features.pt",
                          weights_only=True).numpy()
    with open(DATA_DIR / "processed" / "user_index.json") as f:
        user_index = json.load(f)
    with open(DATA_DIR / "labels" / "final_labels.json") as f:
        labels_raw = json.load(f)

    idx_to_user = {v: k for k, v in user_index.items()}

    # Map labels: 0=oppose, 1=support, -1=unlabelled
    labels = np.full(len(user_index), -1)
    for user, label in labels_raw.items():
        if user in user_index and label is not None:
            labels[user_index[user]] = int(label)

    print(f"Running t-SNE on {features.shape[0]} users ({features.shape[1]}-dim)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    coords = tsne.fit_transform(features)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    unlabelled = labels == -1
    oppose = labels == 0
    support = labels == 1

    ax.scatter(coords[unlabelled, 0], coords[unlabelled, 1],
               c="#CCCCCC", alpha=0.3, s=20, label="Unlabelled", zorder=1)
    ax.scatter(coords[oppose, 0], coords[oppose, 1],
               c="#2196F3", alpha=0.8, s=40, label="Oppose ban (0)",
               edgecolors="white", linewidths=0.5, zorder=2)
    ax.scatter(coords[support, 0], coords[support, 1],
               c="#F44336", alpha=0.8, s=40, label="Support ban (1)",
               edgecolors="white", linewidths=0.5, zorder=2)

    ax.set_title("t-SNE of User SBERT Embeddings (colored by stance)", fontsize=14)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    out_path = RESULTS_DIR / "tsne_embeddings.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved to: {out_path}")
    print(f"  Oppose ban:  {oppose.sum()}")
    print(f"  Support ban: {support.sum()}")
    print(f"  Unlabelled:  {unlabelled.sum()}")


if __name__ == "__main__":
    main()
