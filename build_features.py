"""
Generate per-user SBERT feature vectors from comment text.

For each user, encode every comment with all-MiniLM-L6-v2 (384-dim),
then mean-pool across comments to produce a single feature vector.

Outputs (to data/processed/):
  node_features.pt  – float32 tensor of shape [N_users, 384]
  user_index.json   – ordered mapping {username: row_index}
"""

import json
import os
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent / "data"
CLEAN_DIR = DATA_DIR / "clean"
PROCESSED_DIR = DATA_DIR / "processed"

USER_COMMENTS_PATH = CLEAN_DIR / "clean_user_comments.json"
FEATURES_PATH = PROCESSED_DIR / "node_features.pt"
USER_INDEX_PATH = PROCESSED_DIR / "user_index.json"

MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    with open(USER_COMMENTS_PATH) as f:
        user_comments = json.load(f)

    usernames = sorted(user_comments.keys())
    user_index = {u: i for i, u in enumerate(usernames)}

    print(f"Loading SBERT model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {dim}")
    print(f"Encoding comments for {len(usernames)} users...\n")

    features = torch.zeros(len(usernames), dim)

    for i, username in enumerate(usernames):
        texts = [c["text"] for c in user_comments[username]]
        embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        features[i] = embeddings.mean(dim=0)

        if (i + 1) % 50 == 0 or i == len(usernames) - 1:
            print(f"  [{i+1}/{len(usernames)}] encoded")

    torch.save(features, FEATURES_PATH)
    with open(USER_INDEX_PATH, "w") as f:
        json.dump(user_index, f, indent=2)

    print(f"\nFeature matrix shape: {list(features.shape)}")
    print(f"Saved to: {FEATURES_PATH}")
    print(f"User index saved to: {USER_INDEX_PATH}")


if __name__ == "__main__":
    main()
