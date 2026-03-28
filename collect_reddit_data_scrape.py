"""
Reddit data collection via public JSON endpoints (no API credentials needed).

Produces the same 4 output files as collect_reddit_data.py so the rest of
the pipeline is unaffected. Switch back to the PRAW script once API access
is approved — it handles edge cases more robustly.

Outputs (to data/):
  raw_comments.json     - all comments keyed by comment ID
  edges.csv             - directed user-reply edges (replier → replied-to)
  user_comments.json    - per-user comment texts (for embedding later)
  thread_metadata.json  - per-thread summary info

Requirements:
  pip install requests
"""

import json
import csv
import os
import time
import requests
from collections import defaultdict

# Reddit blocks requests with no User-Agent or a generic one like "python-requests".
# A descriptive string avoids most blocks.
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; comp511_ai_art_scraper/0.1; research use)"
}

THREAD_URLS = [
    "https://www.reddit.com/r/DefendingAIArt/comments/1rroxgz/ai_art_is_now_banned_in_many_places_on_reddit_is/",
    "https://www.reddit.com/r/changemyview/comments/1nym59o/cmv_generative_ai_should_be_banned/",
    "https://www.reddit.com/r/technology/comments/1qjyadw/comiccon_bans_ai_art_after_artist_pushback/",
    "https://www.reddit.com/r/aiwars/comments/1c09qj3/ban_ai_generated_images_and_ai_generators_from/",
    "https://www.reddit.com/r/NoStupidQuestions/comments/zd5ntt/should_ai_art_generators_be_banned/",
    "https://www.reddit.com/r/aiwars/comments/1kqohfx/im_proai_but_subreddits_banning_ai_art_is_totally/",
]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")

# Seconds to wait between requests — be polite, avoid getting temporarily blocked
REQUEST_DELAY = 2


def get_submission_id(url: str) -> str:
    parts = url.rstrip("/").split("/")
    idx = parts.index("comments")
    return parts[idx + 1]


def json_url(url: str) -> str:
    """Convert a Reddit thread URL to its .json equivalent."""
    return url.rstrip("/") + ".json?limit=500&raw_json=1"


def fetch_json(url: str, retries: int = 3) -> dict | list | None:
    """GET a URL and return parsed JSON, with simple retry logic."""
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=15)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"  [rate limit] Waiting {wait}s before retry...")
                time.sleep(wait)
            else:
                print(f"  [warn] HTTP {response.status_code} for {url}")
                return None
        except requests.RequestException as e:
            print(f"  [error] Request failed (attempt {attempt + 1}): {e}")
            time.sleep(5)
    return None


def expand_more_children(link_id: str, children_ids: list[str]) -> list:
    """
    Fetch collapsed comment stubs ("load more comments") via the
    morechildren endpoint. Returns a flat list of comment data dicts.
    """
    if not children_ids:
        return []

    # The endpoint accepts up to ~100 IDs at a time
    all_things = []
    batch_size = 100
    for i in range(0, len(children_ids), batch_size):
        batch = children_ids[i : i + batch_size]
        url = (
            "https://www.reddit.com/api/morechildren.json"
            f"?link_id={link_id}&children={','.join(batch)}"
            "&api_type=json&limit_children=false&raw_json=1"
        )
        time.sleep(REQUEST_DELAY)
        data = fetch_json(url)
        if data and "json" in data and "data" in data["json"]:
            all_things.extend(data["json"]["data"].get("things", []))

    return all_things


def parse_comment(thing_data: dict, submission_id: str) -> dict:
    """Convert a raw comment data dict into our standard comment schema."""
    author = thing_data.get("author") or "[deleted]"
    if author in ("", None):
        author = "[deleted]"

    parent_id_full = thing_data.get("parent_id", f"t3_{submission_id}")
    parent_type = parent_id_full.split("_")[0]
    parent_id   = parent_id_full.split("_")[1]

    return {
        "id":            thing_data["id"],
        "submission_id": submission_id,
        "author":        author,
        "body":          thing_data.get("body", ""),
        "score":         thing_data.get("score", 0),
        "created_utc":   thing_data.get("created_utc", 0),
        "parent_type":   parent_type,
        "parent_id":     parent_id,
        "depth":         thing_data.get("depth", 0),
    }


def flatten_comment_listing(listing: dict, submission_id: str,
                             link_fullname: str, all_comments: dict):
    """
    Recursively walk a comment listing dict and collect every comment.
    Expands 'more' stubs via the morechildren endpoint.
    """
    if not listing or listing.get("kind") != "Listing":
        return

    for child in listing.get("data", {}).get("children", []):
        kind = child.get("kind")
        data = child.get("data", {})

        if kind == "t1":
            # Regular comment
            cid = data.get("id")
            if cid and cid not in all_comments:
                all_comments[cid] = parse_comment(data, submission_id)

            # Recurse into replies
            replies = data.get("replies")
            if replies and isinstance(replies, dict):
                flatten_comment_listing(replies, submission_id,
                                        link_fullname, all_comments)

        elif kind == "more":
            children_ids = data.get("children", [])
            if not children_ids:
                continue
            print(f"    Expanding {len(children_ids)} collapsed comment(s)...")
            things = expand_more_children(link_fullname, children_ids)
            for thing in things:
                if thing.get("kind") == "t1":
                    tdata = thing.get("data", {})
                    cid = tdata.get("id")
                    if cid and cid not in all_comments:
                        all_comments[cid] = parse_comment(tdata, submission_id)


def build_edges(all_comments: dict, thread_meta: dict) -> list[dict]:
    edges = []
    for cid, c in all_comments.items():
        src_user = c["author"]
        if src_user == "[deleted]":
            continue

        if c["parent_type"] == "t1":
            parent = all_comments.get(c["parent_id"])
            if parent is None:
                continue
            dst_user = parent["author"]
            if dst_user == "[deleted]":
                continue
        else:
            sid = c["submission_id"]
            dst_user = thread_meta[sid]["author"]
            if dst_user == "[deleted]":
                continue

        if src_user == dst_user:
            continue

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

    all_comments: dict = {}
    thread_meta:  dict = {}

    for url in THREAD_URLS:
        sid = get_submission_id(url)
        print(f"\nFetching submission {sid} ...")

        data = fetch_json(json_url(url))
        if not data or not isinstance(data, list) or len(data) < 2:
            print(f"  [error] Could not fetch or parse {url}")
            time.sleep(REQUEST_DELAY)
            continue

        post_listing    = data[0]
        comment_listing = data[1]

        # Extract post metadata
        post_children = post_listing.get("data", {}).get("children", [])
        if not post_children:
            print(f"  [error] No post data found for {url}")
            continue

        post_data  = post_children[0].get("data", {})
        op_author  = post_data.get("author") or "[deleted]"
        link_fullname = f"t3_{sid}"

        thread_meta[sid] = {
            "submission_id": sid,
            "url":           url,
            "subreddit":     post_data.get("subreddit", ""),
            "title":         post_data.get("title", ""),
            "author":        op_author,
            "score":         post_data.get("score", 0),
            "created_utc":   post_data.get("created_utc", 0),
            "num_comments":  post_data.get("num_comments", 0),
        }

        before = len(all_comments)
        flatten_comment_listing(comment_listing, sid, link_fullname, all_comments)
        collected = len(all_comments) - before
        print(f"  Collected {collected} comments (total so far: {len(all_comments)})")

        time.sleep(REQUEST_DELAY)

    # Build per-user comment mapping
    user_comments: dict = defaultdict(list)
    for c in all_comments.values():
        if c["author"] != "[deleted]":
            user_comments[c["author"]].append({
                "text":          c["body"],
                "comment_id":    c["id"],
                "submission_id": c["submission_id"],
            })

    # Build edge list
    edges = build_edges(all_comments, thread_meta)

    # Write outputs
    raw_path   = os.path.join(OUTPUT_DIR, "raw_comments.json")
    edges_path = os.path.join(OUTPUT_DIR, "edges.csv")
    users_path = os.path.join(OUTPUT_DIR, "user_comments.json")
    meta_path  = os.path.join(OUTPUT_DIR, "thread_metadata.json")

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

    # Summary
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
