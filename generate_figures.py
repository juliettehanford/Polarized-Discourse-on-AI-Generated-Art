"""
Generate all report-ready figures for the AI art stance benchmark.

Outputs (to results/figures/):
  fig1_label_distribution.png
  fig2_model_comparison.png
  fig3_per_class_precision_recall.png
  fig4_component_sizes.png
  fig5_llm_accuracy.png
  fig6_tsne_embeddings.png
"""

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import networkx as nx
import torch
from sklearn.manifold import TSNE

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"

DPI = 300

CLR_PRO = "#E53935"
CLR_ANTI = "#1E88E5"
CLR_UNLBL = "#BDBDBD"
CLR_ACC = "#5E35B1"
CLR_F1 = "#00897B"
CLR_PREC = "#FB8C00"
CLR_REC = "#43A047"

MODEL_ORDER = ["LabelProp", "MLP", "GCN", "GAT", "GraphSAGE"]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def fig1_label_distribution():
    """Bar chart of pro-ban / anti-ban / unlabelled counts."""
    labels = load_json(DATA_DIR / "labels" / "final_labels.json")
    pro = sum(1 for v in labels.values() if v == 1)
    anti = sum(1 for v in labels.values() if v == 0)
    unlbl = sum(1 for v in labels.values() if v is None)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    categories = ["Pro-ban", "Anti-ban", "Unlabelled"]
    counts = [pro, anti, unlbl]
    colors = [CLR_PRO, CLR_ANTI, CLR_UNLBL]

    bars = ax.bar(categories, counts, color=colors, width=0.55, edgecolor="white",
                  linewidth=1.2)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 4,
                str(count), ha="center", va="bottom", fontsize=13, fontweight="bold")

    ax.set_ylabel("Number of Users", fontsize=12)
    ax.set_title("Label Distribution", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(counts) * 1.15)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_label_distribution.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  fig1_label_distribution.png")


def fig2_model_comparison():
    """Grouped bar chart: accuracy and macro-F1 per model."""
    metrics = load_json(RESULTS_DIR / "metrics.json")

    models = [m for m in MODEL_ORDER if m in metrics]
    acc_mean = [metrics[m]["aggregate"]["accuracy"]["mean"] for m in models]
    acc_std = [metrics[m]["aggregate"]["accuracy"]["std"] for m in models]
    f1_mean = [metrics[m]["aggregate"]["macro_f1"]["mean"] for m in models]
    f1_std = [metrics[m]["aggregate"]["macro_f1"]["std"] for m in models]

    x = np.arange(len(models))
    w = 0.32

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, acc_mean, w, yerr=acc_std, label="Accuracy",
           color=CLR_ACC, capsize=4, edgecolor="white", linewidth=0.8)
    ax.bar(x + w / 2, f1_mean, w, yerr=f1_std, label="Macro-F1",
           color=CLR_F1, capsize=4, edgecolor="white", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison: Accuracy & Macro-F1", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 0.85)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0.5, color="#999999", linewidth=0.8, linestyle="--", zorder=0)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_model_comparison.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  fig2_model_comparison.png")


def fig3_per_class_precision_recall():
    """Side-by-side subplots: precision/recall for each class per model."""
    metrics = load_json(RESULTS_DIR / "metrics.json")
    models = [m for m in MODEL_ORDER if m in metrics]

    def get(model, key):
        agg = metrics[model]["aggregate"]
        return agg[key]["mean"], agg[key]["std"]

    x = np.arange(len(models))
    w = 0.32

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    # Pro-ban (class 1)
    p1_m = [get(m, "class_1_precision")[0] for m in models]
    p1_s = [get(m, "class_1_precision")[1] for m in models]
    r1_m = [get(m, "class_1_recall")[0] for m in models]
    r1_s = [get(m, "class_1_recall")[1] for m in models]

    ax1.bar(x - w / 2, p1_m, w, yerr=p1_s, label="Precision",
            color=CLR_PREC, capsize=4, edgecolor="white", linewidth=0.8)
    ax1.bar(x + w / 2, r1_m, w, yerr=r1_s, label="Recall",
            color=CLR_REC, capsize=4, edgecolor="white", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=10)
    ax1.set_ylabel("Score", fontsize=12)
    ax1.set_title("Pro-ban (Class 1)", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.set_ylim(0, 1.05)

    # Anti-ban (class 0)
    p0_m = [get(m, "class_0_precision")[0] for m in models]
    p0_s = [get(m, "class_0_precision")[1] for m in models]
    r0_m = [get(m, "class_0_recall")[0] for m in models]
    r0_s = [get(m, "class_0_recall")[1] for m in models]

    ax2.bar(x - w / 2, p0_m, w, yerr=p0_s, label="Precision",
            color=CLR_PREC, capsize=4, edgecolor="white", linewidth=0.8)
    ax2.bar(x + w / 2, r0_m, w, yerr=r0_s, label="Recall",
            color=CLR_REC, capsize=4, edgecolor="white", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=10)
    ax2.set_title("Anti-ban (Class 0)", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Per-Class Precision & Recall", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_per_class_precision_recall.png", dpi=DPI,
                bbox_inches="tight")
    plt.close(fig)
    print("  fig3_per_class_precision_recall.png")


def fig4_component_sizes():
    """Bar chart of connected component sizes."""
    stats = load_json(DATA_DIR / "clean" / "graph_stats.json")
    sizes = sorted(stats["component_sizes"], reverse=True)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(sizes))
    bars = ax.bar(x, sizes, color="#5C6BC0", edgecolor="white", linewidth=0.8)

    for i, (bar, s) in enumerate(zip(bars, sizes)):
        if i < 4:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    str(s), ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xlabel("Component Rank", fontsize=12)
    ax.set_ylabel("Number of Users", fontsize=12)
    ax.set_title(f"Connected Component Sizes ({len(sizes)} components, "
                 f"{stats['isolated_users']} isolated users not shown)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(range(0, len(sizes), 2))
    ax.set_xticklabels(range(1, len(sizes) + 1, 2))
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_component_sizes.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  fig4_component_sizes.png")


def fig5_llm_accuracy():
    """LLM annotation accuracy: donut chart + override breakdown bar chart."""
    report = load_json(DATA_DIR / "labels" / "llm_accuracy_report.json")
    agreed = report["llm_agreed"]
    overridden = report["llm_overridden"]
    breakdown = report["override_breakdown"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                   gridspec_kw={"width_ratios": [1, 1.4]})

    # Donut chart
    wedges, texts, autotexts = ax1.pie(
        [agreed, overridden],
        labels=["Agreed", "Overridden"],
        colors=["#66BB6A", "#EF5350"],
        autopct=lambda p: f"{p:.1f}%\n({int(round(p * (agreed + overridden) / 100))})",
        startangle=90,
        pctdistance=0.75,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
        textprops={"fontsize": 11},
    )
    for at in autotexts:
        at.set_fontsize(10)
    ax1.set_title(f"LLM vs Human Agreement\n({agreed + overridden} users reviewed)",
                  fontsize=13, fontweight="bold")

    # Override breakdown
    readable = {
        "support_ban\u2192oppose_ban": "Pro \u2192 Anti",
        "support_ban\u2192ambiguous": "Pro \u2192 Ambig.",
        "oppose_ban\u2192support_ban": "Anti \u2192 Pro",
        "oppose_ban\u2192ambiguous": "Anti \u2192 Ambig.",
        "ambiguous\u2192support_ban": "Ambig. \u2192 Pro",
        "ambiguous\u2192oppose_ban": "Ambig. \u2192 Anti",
    }
    labels_sorted = sorted(breakdown.keys(), key=lambda k: breakdown[k], reverse=True)
    bar_labels = [readable.get(k, k) for k in labels_sorted]
    bar_vals = [breakdown[k] for k in labels_sorted]

    bar_colors = []
    for k in labels_sorted:
        if "ambiguous" in k.split("\u2192")[1] if "\u2192" in k else "":
            bar_colors.append(CLR_UNLBL)
        elif "support" in k.split("\u2192")[1] if "\u2192" in k else "":
            bar_colors.append(CLR_PRO)
        else:
            bar_colors.append(CLR_ANTI)

    bars = ax2.barh(bar_labels[::-1], bar_vals[::-1],
                    color=bar_colors[::-1], edgecolor="white", linewidth=1.0,
                    height=0.55)
    for bar, val in zip(bars, bar_vals[::-1]):
        ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                 str(val), va="center", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Number of Overrides", fontsize=12)
    ax2.set_title("Override Breakdown by Direction", fontsize=13, fontweight="bold")
    ax2.set_xlim(0, max(bar_vals) * 1.25)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_llm_accuracy.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  fig5_llm_accuracy.png")


def fig6_tsne():
    """t-SNE of SBERT embeddings colored by stance."""
    features = torch.load(DATA_DIR / "processed" / "node_features.pt",
                          weights_only=True).numpy()
    user_index = load_json(DATA_DIR / "processed" / "user_index.json")
    labels_raw = load_json(DATA_DIR / "labels" / "final_labels.json")

    labels = np.full(len(user_index), -1)
    for user, label in labels_raw.items():
        if user in user_index and label is not None:
            labels[user_index[user]] = int(label)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    coords = tsne.fit_transform(features)

    unlabelled = labels == -1
    oppose = labels == 0
    support = labels == 1

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(coords[unlabelled, 0], coords[unlabelled, 1],
               c=CLR_UNLBL, alpha=0.3, s=20, label="Unlabelled", zorder=1)
    ax.scatter(coords[oppose, 0], coords[oppose, 1],
               c=CLR_ANTI, alpha=0.8, s=40, label="Anti-ban",
               edgecolors="white", linewidths=0.5, zorder=2)
    ax.scatter(coords[support, 0], coords[support, 1],
               c=CLR_PRO, alpha=0.8, s=40, label="Pro-ban",
               edgecolors="white", linewidths=0.5, zorder=2)

    ax.set_title("t-SNE of User SBERT Embeddings (colored by stance)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_tsne_embeddings.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  fig6_tsne_embeddings.png")


def fig7_directed_graph():
    """Network visualization of the full directed reply graph."""
    user_index = load_json(DATA_DIR / "processed" / "user_index.json")
    labels_raw = load_json(DATA_DIR / "labels" / "final_labels.json")

    G = nx.DiGraph()
    G.add_nodes_from(range(len(user_index)))

    with open(DATA_DIR / "clean" / "clean_edges.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s, d = row["src_user"], row["dst_user"]
            if s in user_index and d in user_index:
                G.add_edge(user_index[s], user_index[d])

    labels = {}
    for user, lbl in labels_raw.items():
        if user in user_index:
            labels[user_index[user]] = lbl

    node_colors = []
    for n in G.nodes():
        lbl = labels.get(n)
        if lbl == 1:
            node_colors.append(CLR_PRO)
        elif lbl == 0:
            node_colors.append(CLR_ANTI)
        else:
            node_colors.append(CLR_UNLBL)

    degrees = np.array([G.degree(n) for n in G.nodes()])
    node_sizes = 15 + 60 * (degrees / max(degrees.max(), 1))

    # Layout: spring layout per connected component, then pack them together
    components = list(nx.weakly_connected_components(G))
    components.sort(key=len, reverse=True)

    pos = {}
    x_offset = 0
    for comp in components:
        sub = G.subgraph(comp)
        sub_pos = nx.spring_layout(sub, k=1.8 / (len(comp) ** 0.4),
                                   iterations=120, seed=42)
        # Normalise to unit square
        xs = np.array([sub_pos[n][0] for n in comp])
        ys = np.array([sub_pos[n][1] for n in comp])
        span = max(np.ptp(xs), np.ptp(ys), 1e-9)
        scale = (len(comp) / len(user_index)) ** 0.5  # proportional area
        for n in comp:
            pos[n] = ((sub_pos[n][0] - xs.min()) / span * scale + x_offset,
                      (sub_pos[n][1] - ys.min()) / span * scale)
        x_offset += scale + 0.08

    fig, ax = plt.subplots(figsize=(16, 10))

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color="#CCCCCC", alpha=0.35, width=0.4,
        arrows=True, arrowsize=4, arrowstyle="-|>",
        connectionstyle="arc3,rad=0.08",
        min_source_margin=2, min_target_margin=2,
    )

    # Draw unlabelled first (background), then labelled on top
    for lbl_val, color, label_text, alpha, zorder in [
        (-1, CLR_UNLBL, "Unlabelled", 0.4, 1),
        (0, CLR_ANTI, "Anti-ban", 0.9, 2),
        (1, CLR_PRO, "Pro-ban", 0.9, 2),
    ]:
        if lbl_val == -1:
            nodelist = [n for n in G.nodes() if labels.get(n) is None]
        else:
            nodelist = [n for n in G.nodes() if labels.get(n) == lbl_val]
        if not nodelist:
            continue
        sizes = [node_sizes[n] for n in nodelist]
        nx.draw_networkx_nodes(
            G, pos, nodelist=nodelist, ax=ax,
            node_color=color, node_size=sizes,
            alpha=alpha, edgecolors="white", linewidths=0.3,
            label=label_text,
        )

    ax.set_title("Directed Reply Graph (439 users, 787 edges, 5 components)",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=11,
              markerscale=1.5)
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7_directed_graph.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  fig7_directed_graph.png")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating report figures...")

    fig1_label_distribution()
    fig2_model_comparison()
    fig3_per_class_precision_recall()
    fig4_component_sizes()
    fig5_llm_accuracy()
    fig6_tsne()
    fig7_directed_graph()

    print(f"\nAll figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
