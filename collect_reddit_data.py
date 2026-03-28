"""
Reddit data collection for:
  "Polarized Discourse on AI-Generated Art: Constructing a Graph Benchmark
   from Reddit for Stance Classification"

Collects comment text, reply chains, and user interactions from target threads.
Outputs:
  data/raw_comments.json     - all comments keyed by comment ID
  data/edges.csv             - directed user-reply edges (replier → replied-to)
  data/user_comments.json    - per-user comment texts (for embedding later)
  data/thread_metadata.json  - per-thread summary info

Requirements:
  pip install praw

Reddit API credentials:
  1. Go to https://www.reddit.com/prefs/apps
  2. Click "create another app" → select "script"
  3. Fill in name/description, set redirect URI to http://localhost:8080
  4. Copy the client_id (under app name) and client_secret
  5. Set the three variables below (or use environment variables)
"""

import json
import csv
import os
import time
import praw
from collections import defaultdict

# ── Credentials ──────────────────────────────────────────────────────────────
CLIENT_ID     = os.getenv("REDDIT_CLIENT_ID",     "YOUR_CLIENT_ID")
CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "YOUR_CLIENT_SECRET")
USER_AGENT    = os.getenv("REDDIT_USER_AGENT",    "comp511_ai_art_scraper/0.1 (by u/YOUR_USERNAME)")
# ─────────────────────────────────────────────────────────────────────────────

THREAD_URLS = [
    "https://www.reddit.com/r/DefendingAIArt/comments/1rroxgz/ai_art_is_now_banned_in_many_places_on_reddit_is/",
    "https://www.reddit.com/r/changemyview/comments/1nym59o/cmv_generative_ai_should_be_banned/",
    "https://www.reddit.com/r/technology/comments/1qjyadw/comiccon_bans_ai_art_after_artist_pushback/",
    "https://www.reddit.com/r/aiwars/comments/1c09qj3/ban_ai_generated_images_and_ai_generators_from/",
    "https://www.reddit.com/r/NoStupidQuestions/comments/zd5ntt/should_ai_art_generators_be_banned/",
    "https://www.reddit.com/r/aiwars/comments/1kqohfx/im_proai_but_subreddits_banning_ai_art_is_totally/",
]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "project")


def get_submission_id(url: str) -> str:
    """Extract the submission ID from a Reddit URL."""
    # URLs look like: .../comments/<id>/...
    parts = url.rstrip("/").split("/")
    idx = parts.index("comments")
    return parts[idx + 1]


def flatten_comments(comment_forest, submission_id: str, all_comments: dict):
    """
    Recursively walk the CommentForest and collect every comment.
    Stores each comment as a dict with fields needed to reconstruct reply chains.

    all_comments: {comment_id -> comment_dict}  (mutated in place)
    """
    for comment in comment_forest:
        if isinstance(comment, praw.models.MoreComments):
            # Expand "load more comments" up to a reasonable limit
            try:
                more = comment.comments()
                flatten_comments(more, submission_id, all_comments)
            except Exception as e:
                print(f"  [warn] Could not expand MoreComments: {e}")
            continue

        author = str(comment.author) if comment.author else "[deleted]"

        # parent_id is "t1_<comment_id>" for comment replies
        # or "t3_<submission_id>" for top-level comments
        parent_id_full = comment.parent_id          # e.g. "t1_abc123" or "t3_xyz"
        parent_type    = parent_id_full.split("_")[0]  # "t1" or "t3"
        parent_id      = parent_id_full.split("_")[1]

        all_comments[comment.id] = {
            "id":            comment.id,
            "submission_id": submission_id,
            "author":        author,
            "body":          comment.body,
            "score":         comment.score,
            "created_utc":   comment.created_utc,
            "parent_type":   parent_type,   # "t1" = comment, "t3" = post
            "parent_id":     parent_id,     # bare ID (no prefix)
            "depth":         comment.depth,
        }

        # Recurse into replies
        if comment.replies:
            flatten_comments(comment.replies, submission_id, all_comments)


def build_edges(all_comments: dict, thread_meta: dict) -> list[dict]:
    """
    Build directed user-reply edges.

    Edge (u → v) means: user u replied to a comment/post by user v.

    For top-level comments (parent_type == "t3"), the target is the OP of that
    submission, retrieved from thread_meta.
    """
    edges = []

    for cid, c in all_comments.items():
        src_user = c["author"]
        if src_user == "[deleted]":
            continue

        if c["parent_type"] == "t1":
            # Reply to another comment
            parent = all_comments.get(c["parent_id"])
            if parent is None:
                continue   # parent was not collected (pruned / deleted)
            dst_user = parent["author"]
            if dst_user == "[deleted]":
                continue
        else:
            # Top-level reply to the submission → target is the OP
            sid = c["submission_id"]
            dst_user = thread_meta[sid]["author"]
            if dst_user == "[deleted]":
                continue

        if src_user == dst_user:
            continue  # skip self-loops

        edges.append({
            "src_user":      src_user,
            "dst_user":      dst_user,
            "comment_id":    cid,
            "parent_id":     c["parent_id"],
            "submission_id": c["submission_id"],
            "subreddit":     thread_meta[c["submission_id"]]["subreddit"],
        })

    return edges


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USER_AGENT,
    )

    all_comments: dict  = {}   # comment_id → comment dict
    thread_meta:  dict  = {}   # submission_id → metadata

    for url in THREAD_URLS:
        sid = get_submission_id(url)
        print(f"\nFetching submission {sid} ...")

        try:
            submission = reddit.submission(id=sid)
            # Expand all "MoreComments" in the forest (limit=None = fetch all)
            submission.comments.replace_more(limit=None)
        except Exception as e:
            print(f"  [error] Could not fetch {url}: {e}")
            continue

        op_author = str(submission.author) if submission.author else "[deleted]"

        thread_meta[sid] = {
            "submission_id": sid,
            "url":           url,
            "subreddit":     str(submission.subreddit),
            "title":         submission.title,
            "author":        op_author,
            "score":         submission.score,
            "created_utc":   submission.created_utc,
            "num_comments":  submission.num_comments,
        }

        before = len(all_comments)
        flatten_comments(submission.comments, sid, all_comments)
        collected = len(all_comments) - before
        print(f"  Collected {collected} comments (total so far: {len(all_comments)})")

        # Be polite to the API
        time.sleep(1)

    # ── Derive user → [comments] mapping ────────────────────────────────────
    user_comments: dict = defaultdict(list)
    for c in all_comments.values():
        if c["author"] != "[deleted]":
            user_comments[c["author"]].append({
                "text":          c["body"],
                "comment_id":    c["id"],
                "submission_id": c["submission_id"],
            })

    # ── Build edge list ──────────────────────────────────────────────────────
    edges = build_edges(all_comments, thread_meta)

    # ── Write outputs ────────────────────────────────────────────────────────
    raw_path      = os.path.join(OUTPUT_DIR, "raw_comments.json")
    edges_path    = os.path.join(OUTPUT_DIR, "edges.csv")
    users_path    = os.path.join(OUTPUT_DIR, "user_comments.json")
    meta_path     = os.path.join(OUTPUT_DIR, "thread_metadata.json")

    with open(raw_path, "w") as f:
        json.dump(all_comments, f, indent=2)

    with open(edges_path, "w", newline="") as f:
        fieldnames = ["src_user", "dst_user", "comment_id", "parent_id",
                      "submission_id", "subreddit"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(edges)

    with open(users_path, "w") as f:
        json.dump(dict(user_comments), f, indent=2)

    with open(meta_path, "w") as f:
        json.dump(thread_meta, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────────────
    unique_users = len(user_comments)
    print(f"\n{'='*50}")
    print(f"Threads collected : {len(thread_meta)}")
    print(f"Total comments    : {len(all_comments)}")
    print(f"Unique users      : {unique_users}")
    print(f"Directed edges    : {len(edges)}")
    print(f"\nOutputs written to: {OUTPUT_DIR}/")
    print(f"  {os.path.basename(raw_path)}")
    print(f"  {os.path.basename(edges_path)}")
    print(f"  {os.path.basename(users_path)}")
    print(f"  {os.path.basename(meta_path)}")


if __name__ == "__main__":
    main()
