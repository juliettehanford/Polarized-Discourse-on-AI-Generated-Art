"""
LLM-assisted stance labelling for the AI art ban debate.

Modes:
  python label_users.py label     -- call OpenAI API to pre-label all users
  python label_users.py finalize  -- convert reviewed CSV to final_labels.json

Set OPENAI_API_KEY via environment variable or in a .env file in the project root.
"""

import json
import csv
import os
import sys
import time
from pathlib import Path

# Load .env file if present (before importing openai)
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip("'\""))

from openai import OpenAI

DATA_DIR = Path(__file__).parent / "data"
CLEAN_DIR = DATA_DIR / "clean"
LABELS_DIR = DATA_DIR / "labels"

THREAD_META_PATH = DATA_DIR / "thread_metadata.json"
USER_COMMENTS_PATH = CLEAN_DIR / "clean_user_comments.json"

RAW_LABELS_PATH = LABELS_DIR / "llm_raw_labels.json"
REVIEW_CSV_PATH = LABELS_DIR / "labels_for_review.csv"
FINAL_LABELS_PATH = LABELS_DIR / "final_labels.json"

MODEL = "gpt-4o-mini"
BATCH_SIZE = 10  # users per API call to stay under RPM limits
REQUEST_DELAY = 22  # seconds between requests (3 RPM limit = 1 per 20s + buffer)

SYSTEM_PROMPT = """\
You are a research assistant labelling Reddit users' stances in an online debate \
about whether AI-generated art should be banned.

For each user you will receive their comments (with the thread title for context). \
Classify each user's overall stance as one of:
  - "support_ban"  : the user favours banning or restricting AI-generated art
  - "oppose_ban"   : the user opposes banning AI-generated art / defends AI art
  - "ambiguous"    : the stance is unclear, neutral, off-topic, or mixed

Return ONLY a valid JSON object mapping each username to their classification:
{
  "username1": {"stance": "oppose_ban", "confidence": 0.9, "reasoning": "..."},
  "username2": {"stance": "support_ban", "confidence": 0.8, "reasoning": "..."},
  ...
}
"""


def build_batch_prompt(batch: list[tuple[str, list[dict]]], thread_meta: dict) -> str:
    """Format a batch of users' comments into a single prompt."""
    parts = []
    for username, comments in batch:
        parts.append(f"=== User: {username} ===")
        by_thread: dict[str, list[str]] = {}
        for c in comments:
            sid = c["submission_id"]
            by_thread.setdefault(sid, []).append(c["text"])

        for sid, texts in by_thread.items():
            title = thread_meta.get(sid, {}).get("title", sid)
            parts.append(f'Thread: "{title}"')
            for i, t in enumerate(texts, 1):
                parts.append(f"  Comment {i}: {t.strip()}")
        parts.append("")
    return "\n".join(parts)


def call_api_with_retry(client, prompt: str, max_retries: int = 5) -> dict:
    """Call the OpenAI API with exponential backoff on rate limit errors."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=2000,
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate_limit" in error_str.lower():
                wait = REQUEST_DELAY * (attempt + 1)
                print(f"    Rate limited, waiting {wait}s (attempt {attempt+1})...")
                time.sleep(wait)
            else:
                print(f"    API error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    raise
    return {}


def label_all_users():
    """Call OpenAI API in batches and save raw results + review CSV."""
    client = OpenAI()

    with open(USER_COMMENTS_PATH) as f:
        user_comments = json.load(f)
    with open(THREAD_META_PATH) as f:
        thread_meta = json.load(f)

    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    # Resume support: skip users already labelled
    existing = {}
    if RAW_LABELS_PATH.exists():
        with open(RAW_LABELS_PATH) as f:
            existing = json.load(f)
        print(f"Resuming -- {len(existing)} users already labelled")

    raw_labels = dict(existing)
    users = sorted(user_comments.keys())
    remaining = [(u, user_comments[u]) for u in users if u not in raw_labels]
    total = len(users)

    # Split into batches
    batches = []
    for i in range(0, len(remaining), BATCH_SIZE):
        batches.append(remaining[i:i + BATCH_SIZE])

    print(f"Labelling {len(remaining)} users in {len(batches)} batches "
          f"(batch size {BATCH_SIZE}, ~{REQUEST_DELAY}s between requests)\n")

    labelled_so_far = len(existing)
    for batch_idx, batch in enumerate(batches):
        batch_usernames = [u for u, _ in batch]
        prompt = build_batch_prompt(batch, thread_meta)

        print(f"  Batch {batch_idx+1}/{len(batches)} "
              f"({len(batch)} users: {batch_usernames[0]}...{batch_usernames[-1]})")

        try:
            result = call_api_with_retry(client, prompt)
        except Exception as e:
            print(f"    FAILED after retries: {e}")
            for username, _ in batch:
                raw_labels[username] = {
                    "stance": "ambiguous", "confidence": 0.0,
                    "reasoning": f"API error: {e}"
                }
            continue

        for username, _ in batch:
            if username in result:
                raw_labels[username] = result[username]
            else:
                raw_labels[username] = {
                    "stance": "ambiguous", "confidence": 0.0,
                    "reasoning": "Not returned by API in batch response"
                }
            labelled_so_far += 1

        # Print batch results
        for username, _ in batch:
            r = raw_labels[username]
            print(f"    [{labelled_so_far}/{total}] {username:30s} -> "
                  f"{r.get('stance', '?'):12s} (conf={r.get('confidence', '?')})")

        # Save after each batch
        with open(RAW_LABELS_PATH, "w") as f:
            json.dump(raw_labels, f, indent=2)

        # Rate limit delay (skip after last batch)
        if batch_idx < len(batches) - 1:
            print(f"    Waiting {REQUEST_DELAY}s for rate limit...")
            time.sleep(REQUEST_DELAY)

    print(f"\nRaw labels saved to {RAW_LABELS_PATH}")

    # Build review CSV sorted by confidence (low first)
    rows = []
    for username in users:
        r = raw_labels.get(username, {})
        sample = user_comments[username][0]["text"][:200]
        rows.append({
            "user": username,
            "llm_stance": r.get("stance", "ambiguous"),
            "confidence": r.get("confidence", 0.0),
            "reasoning": r.get("reasoning", ""),
            "sample_comment": sample,
            "reviewed_stance": "",
        })
    rows.sort(key=lambda r: r["confidence"])

    with open(REVIEW_CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    stances = [r.get("stance") for r in raw_labels.values()]
    print(f"\nLabel distribution:")
    for s in ["oppose_ban", "support_ban", "ambiguous"]:
        print(f"  {s}: {stances.count(s)}")
    print(f"\nReview CSV saved to {REVIEW_CSV_PATH}")
    print(f"  -> Edit 'reviewed_stance' column for corrections (blank = accept LLM)")
    print(f"  -> Then run: python label_users.py finalize")


def _parse_txt_overrides(path: Path, field: str) -> dict[str, str]:
    """Parse a review .txt file and return {username: label} for filled entries."""
    import re
    overrides = {}
    current_user = None
    for line in path.read_text().splitlines():
        m = re.match(r"^--- #\d+\s+(\S+)\s+\(", line)
        if m:
            current_user = m.group(1)
            continue
        if line.startswith(f"{field}:") and current_user:
            label = line.split(":", 1)[1].strip()
            if label:
                overrides[current_user] = label
            current_user = None
    return overrides


def finalize_labels():
    """Convert the reviewed CSV + manual overrides into final_labels.json."""
    if not REVIEW_CSV_PATH.exists():
        print(f"ERROR: {REVIEW_CSV_PATH} not found. Run 'label' first.")
        sys.exit(1)

    stance_map = {"support_ban": 1, "oppose_ban": 0, "ambiguous": None}

    # Load raw LLM labels for accuracy comparison later
    llm_labels = {}
    if RAW_LABELS_PATH.exists():
        with open(RAW_LABELS_PATH) as f:
            for user, r in json.load(f).items():
                llm_labels[user] = r.get("stance", "ambiguous")

    # Start from the main review CSV
    final = {}
    with open(REVIEW_CSV_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            reviewed = row.get("reviewed_stance", "").strip()
            stance = reviewed if reviewed else row["llm_stance"]
            final[row["user"]] = stance_map.get(stance)

    # Apply overrides from the TSV review file
    tsv_path = LABELS_DIR / "review_ambiguous.tsv"
    tsv_overrides = 0
    if tsv_path.exists():
        with open(tsv_path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                label = row.get("your_label", "").strip()
                if label and label in stance_map and row["user"] in final:
                    final[row["user"]] = stance_map[label]
                    tsv_overrides += 1
        if tsv_overrides:
            print(f"Applied {tsv_overrides} overrides from review_ambiguous.tsv")

    # Apply overrides from the plain-text review files
    review_files = {
        "review_ambiguous.txt": "LABEL",
        "review_support_ban.txt": "OVERRIDE",
        "review_oppose_ban.txt": "OVERRIDE",
    }
    all_overrides = {}  # user -> human label (text), across all files
    for filename, field in review_files.items():
        fpath = LABELS_DIR / filename
        if not fpath.exists():
            continue
        file_overrides = _parse_txt_overrides(fpath, field)
        for user, label in file_overrides.items():
            if label in stance_map and user in final:
                final[user] = stance_map[label]
                all_overrides[user] = label
        if file_overrides:
            print(f"Applied {len(file_overrides)} overrides from {filename}")

    # Final label counts
    counts = {"support_ban": 0, "oppose_ban": 0, "ambiguous": 0}
    for v in final.values():
        if v == 1: counts["support_ban"] += 1
        elif v == 0: counts["oppose_ban"] += 1
        else: counts["ambiguous"] += 1

    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(FINAL_LABELS_PATH, "w") as f:
        json.dump(final, f, indent=2)

    labelled = sum(1 for v in final.values() if v is not None)
    print(f"\nFinal labels written to {FINAL_LABELS_PATH}")
    print(f"  support_ban (1): {counts['support_ban']}")
    print(f"  oppose_ban  (0): {counts['oppose_ban']}")
    print(f"  ambiguous (null): {counts['ambiguous']}")
    print(f"  Total labelled:  {labelled} / {len(final)}")

    # ── LLM Accuracy Report ─────────────────────────────────────────────
    # Compare LLM labels against final human-reviewed labels to measure
    # how often the human reviewer agreed with the LLM.
    reviewed_users = set()  # users whose review file was filled in (any file)
    # From support_ban / oppose_ban files: every user in the file was reviewed,
    # but only those with a non-blank OVERRIDE disagreed.
    for filename, field in review_files.items():
        fpath = LABELS_DIR / filename
        if not fpath.exists():
            continue
        import re
        current_user = None
        for line in fpath.read_text().splitlines():
            m = re.match(r"^--- #\d+\s+(\S+)\s+\(", line)
            if m:
                current_user = m.group(1)
                continue
            if line.startswith(f"{field}:") and current_user:
                # User was reviewed (regardless of whether they were overridden)
                reviewed_users.add(current_user)
                current_user = None

    if not reviewed_users:
        print("\n  (No manual reviews detected — skipping accuracy report)")
        return

    # For each reviewed user, compare LLM stance to final stance
    agree = 0
    disagree = 0
    disagree_details = {"support_ban→oppose_ban": 0, "support_ban→ambiguous": 0,
                        "oppose_ban→support_ban": 0, "oppose_ban→ambiguous": 0,
                        "ambiguous→support_ban": 0, "ambiguous→oppose_ban": 0}
    label_name = {1: "support_ban", 0: "oppose_ban", None: "ambiguous"}

    for user in reviewed_users:
        llm = llm_labels.get(user, "ambiguous")
        final_val = final.get(user)
        final_str = label_name[final_val]
        if llm == final_str:
            agree += 1
        else:
            disagree += 1
            key = f"{llm}→{final_str}"
            disagree_details[key] = disagree_details.get(key, 0) + 1

    total_reviewed = agree + disagree
    accuracy = agree / total_reviewed if total_reviewed > 0 else 0

    report = {
        "total_users": len(final),
        "users_reviewed": total_reviewed,
        "llm_agreed": agree,
        "llm_overridden": disagree,
        "llm_accuracy": round(accuracy, 4),
        "override_breakdown": {k: v for k, v in disagree_details.items() if v > 0},
    }

    report_path = LABELS_DIR / "llm_accuracy_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*50}")
    print(f"LLM ANNOTATION ACCURACY REPORT")
    print(f"{'='*50}")
    print(f"  Users reviewed by human:  {total_reviewed}")
    print(f"  LLM agreed with human:    {agree} ({100*accuracy:.1f}%)")
    print(f"  Human overrode LLM:       {disagree} ({100*(1-accuracy):.1f}%)")
    if disagree > 0:
        print(f"  Override breakdown:")
        for change, count in sorted(disagree_details.items()):
            if count > 0:
                print(f"    {change}: {count}")
    print(f"\n  Report saved to: {report_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python label_users.py [label|finalize]")
        sys.exit(1)

    cmd = sys.argv[1].lower()
    if cmd == "label":
        label_all_users()
    elif cmd == "finalize":
        finalize_labels()
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python label_users.py [label|finalize]")
        sys.exit(1)
