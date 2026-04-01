"""
Construct a PyTorch Geometric Data object from processed features, edges, and labels.

Inputs:
  data/clean/clean_edges.csv
  data/processed/node_features.pt  +  user_index.json
  data/labels/final_labels.json

Output:
  data/processed/ai_art_stance.pt  – a torch_geometric.data.Data object with:
    x            [N, 384]       node feature matrix
    edge_index   [2, E]         directed edges
    y            [N]            labels (0=oppose_ban, 1=support_ban, -1=unlabelled)
    train_mask   [N]            boolean
    val_mask     [N]            boolean
    test_mask    [N]            boolean
"""

import csv
import json
from collections import defaultdict, deque
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).parent / "data"
CLEAN_DIR = DATA_DIR / "clean"
PROCESSED_DIR = DATA_DIR / "processed"
LABELS_DIR = DATA_DIR / "labels"

EDGES_PATH = CLEAN_DIR / "clean_edges.csv"
FEATURES_PATH = PROCESSED_DIR / "node_features.pt"
USER_INDEX_PATH = PROCESSED_DIR / "user_index.json"
LABELS_PATH = LABELS_DIR / "final_labels.json"
OUTPUT_PATH = PROCESSED_DIR / "ai_art_stance.pt"


def find_components(adj: dict, nodes: set) -> dict:
    """Return a dict mapping each node to its component ID."""
    visited = {}
    comp_id = 0
    for start in nodes:
        if start in visited:
            continue
        q = deque([start])
        while q:
            node = q.popleft()
            if node in visited:
                continue
            visited[node] = comp_id
            for nb in adj.get(node, set()):
                if nb not in visited:
                    q.append(nb)
        comp_id += 1
    return visited


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Load inputs
    features = torch.load(FEATURES_PATH, weights_only=True)
    with open(USER_INDEX_PATH) as f:
        user_index = json.load(f)
    with open(LABELS_PATH) as f:
        labels_raw = json.load(f)

    n_users = len(user_index)
    idx_to_user = {v: k for k, v in user_index.items()}

    # Build edge_index
    edges_src, edges_dst = [], []
    adj = defaultdict(set)
    with open(EDGES_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            s, d = row["src_user"], row["dst_user"]
            if s in user_index and d in user_index:
                si, di = user_index[s], user_index[d]
                edges_src.append(si)
                edges_dst.append(di)
                adj[si].add(di)
                adj[di].add(si)

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)

    # Labels: 0 = oppose_ban, 1 = support_ban, -1 = unlabelled/ambiguous
    y = torch.full((n_users,), -1, dtype=torch.long)
    for username, label in labels_raw.items():
        if username in user_index and label is not None:
            y[user_index[username]] = int(label)

    labelled_mask = y >= 0
    labelled_indices = labelled_mask.nonzero(as_tuple=True)[0].tolist()
    labelled_labels = y[labelled_indices].tolist()

    print(f"Total nodes:     {n_users}")
    print(f"Total edges:     {edge_index.shape[1]}")
    print(f"Labelled nodes:  {len(labelled_indices)}")
    print(f"  oppose_ban (0): {labelled_labels.count(0)}")
    print(f"  support_ban(1): {labelled_labels.count(1)}")
    print(f"Unlabelled:      {n_users - len(labelled_indices)}")

    # Compute component IDs for stratification
    node_components = find_components(adj, set(range(n_users)))
    comp_labels = [node_components.get(i, -1) for i in labelled_indices]
    # Combine (component, label) for stratification key
    strat_key = [f"{c}_{l}" for c, l in zip(comp_labels, labelled_labels)]

    # Stratified 60/20/20 split
    # First split: 60% train, 40% temp
    # Handle case where some strata are too small for stratification
    try:
        train_idx, temp_idx, train_strat, temp_strat = train_test_split(
            labelled_indices, strat_key, test_size=0.4, random_state=42,
            stratify=strat_key
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=42,
            stratify=temp_strat
        )
    except ValueError:
        # Fall back to label-only stratification if component+label bins too small
        print("  (falling back to label-only stratification — some components too small)")
        train_idx, temp_idx, _, temp_labels = train_test_split(
            labelled_indices, labelled_labels, test_size=0.4, random_state=42,
            stratify=labelled_labels
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=42,
            stratify=temp_labels
        )

    train_mask = torch.zeros(n_users, dtype=torch.bool)
    val_mask = torch.zeros(n_users, dtype=torch.bool)
    test_mask = torch.zeros(n_users, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    print(f"\nSplit sizes:")
    print(f"  Train: {train_mask.sum().item()}")
    print(f"  Val:   {val_mask.sum().item()}")
    print(f"  Test:  {test_mask.sum().item()}")

    # Assemble PyG-compatible Data dict
    # Saved as a plain dict so torch_geometric is not required at build time
    data = {
        "x": features,
        "edge_index": edge_index,
        "y": y,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "num_classes": 2,
        "user_index": user_index,
    }

    torch.save(data, OUTPUT_PATH)
    print(f"\nGraph saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
