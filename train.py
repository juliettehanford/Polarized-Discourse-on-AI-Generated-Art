"""
Train and evaluate all models on the AI art stance graph benchmark.

Usage:
  python train.py                   # run all models, 10 seeds
  python train.py --seeds 5         # fewer seeds for quick test
  python train.py --models MLP GCN  # run subset of models

Outputs (to results/):
  metrics.json        – per-model, per-seed results
  comparison_table.txt – formatted summary table
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_recall_fscore_support)
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from models import MLP, GCN, GAT, GraphSAGE

DATA_PATH = Path(__file__).parent / "data" / "processed" / "ai_art_stance.pt"
RESULTS_DIR = Path(__file__).parent / "results"

MODEL_REGISTRY = {
    "MLP": MLP,
    "GCN": GCN,
    "GAT": GAT,
    "GraphSAGE": GraphSAGE,
}

HIDDEN = 64
LR = 0.01
WEIGHT_DECAY = 5e-4
EPOCHS = 500
PATIENCE = 50


def load_data() -> Data:
    raw = torch.load(DATA_PATH, weights_only=False)

    # GCN/GAT/SAGE expect undirected edges for symmetric message passing
    edge_index_undir = to_undirected(raw["edge_index"])

    data = Data(
        x=raw["x"],
        edge_index=edge_index_undir,
        y=raw["y"],
        train_mask=raw["train_mask"],
        val_mask=raw["val_mask"],
        test_mask=raw["test_mask"],
    )
    data.num_classes = raw["num_classes"]
    return data


def train_epoch(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    preds = out[mask].argmax(dim=1).cpu().numpy()
    labels = data.y[mask].cpu().numpy()

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=[0, 1], zero_division=0
    )

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "class_0_precision": round(prec[0], 4),
        "class_0_recall": round(rec[0], 4),
        "class_0_f1": round(f1[0], 4),
        "class_1_precision": round(prec[1], 4),
        "class_1_recall": round(rec[1], 4),
        "class_1_f1": round(f1[1], 4),
    }


def run_single_seed(model_name: str, data: Data, seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model_cls = MODEL_REGISTRY[model_name]
    in_channels = data.x.shape[1]
    model = model_cls(in_channels, HIDDEN, data.num_classes)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_f1 = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, data, optimizer)
        val_metrics = evaluate(model, data, data.val_mask)

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    # Restore best model and evaluate on test set
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, data, data.test_mask)
    val_metrics = evaluate(model, data, data.val_mask)

    return {
        "seed": seed,
        "best_epoch": epoch - patience_counter,
        "val": val_metrics,
        "test": test_metrics,
    }


def aggregate_results(seed_results: list[dict]) -> dict:
    """Compute mean +/- std across seeds for each test metric."""
    metric_keys = seed_results[0]["test"].keys()
    agg = {}
    for k in metric_keys:
        vals = [r["test"][k] for r in seed_results]
        agg[k] = {
            "mean": round(np.mean(vals), 4),
            "std": round(np.std(vals), 4),
        }
    return agg


def format_table(all_results: dict) -> str:
    """Produce a text comparison table."""
    header = f"{'Model':<12} {'Accuracy':>14} {'Macro-F1':>14} {'P(ban)':>14} {'R(ban)':>14} {'P(no-ban)':>14} {'R(no-ban)':>14}"
    sep = "-" * len(header)
    lines = [sep, header, sep]

    for model_name, res in all_results.items():
        agg = res["aggregate"]

        def fmt(key):
            return f"{agg[key]['mean']:.3f}±{agg[key]['std']:.3f}"

        lines.append(
            f"{model_name:<12} {fmt('accuracy'):>14} {fmt('macro_f1'):>14} "
            f"{fmt('class_1_precision'):>14} {fmt('class_1_recall'):>14} "
            f"{fmt('class_0_precision'):>14} {fmt('class_0_recall'):>14}"
        )
    lines.append(sep)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Train stance classification models")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY.keys()))
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()
    print(f"Graph loaded: {data.num_nodes} nodes, {data.edge_index.shape[1]} edges")
    print(f"Train/Val/Test: {data.train_mask.sum()}/{data.val_mask.sum()}/{data.test_mask.sum()}")
    print(f"Classes: {data.num_classes}\n")

    all_results = {}

    for model_name in args.models:
        if model_name not in MODEL_REGISTRY:
            print(f"Unknown model: {model_name}, skipping")
            continue

        print(f"{'='*50}")
        print(f"Training {model_name} ({args.seeds} seeds)")
        print(f"{'='*50}")

        seed_results = []
        for seed in range(args.seeds):
            result = run_single_seed(model_name, data, seed)
            seed_results.append(result)
            t = result["test"]
            print(f"  Seed {seed}: Acc={t['accuracy']:.3f}  F1={t['macro_f1']:.3f}  "
                  f"(stopped at epoch {result['best_epoch']})")

        agg = aggregate_results(seed_results)
        all_results[model_name] = {
            "seeds": seed_results,
            "aggregate": agg,
        }
        print(f"  Mean test Acc: {agg['accuracy']['mean']:.3f} ± {agg['accuracy']['std']:.3f}")
        print(f"  Mean test F1:  {agg['macro_f1']['mean']:.3f} ± {agg['macro_f1']['std']:.3f}")
        print()

    # Save results
    metrics_path = RESULTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Detailed metrics saved to: {metrics_path}")

    # Print and save comparison table
    table = format_table(all_results)
    table_path = RESULTS_DIR / "comparison_table.txt"
    with open(table_path, "w") as f:
        f.write(table + "\n")
    print(f"\n{table}")
    print(f"\nTable saved to: {table_path}")


if __name__ == "__main__":
    main()
