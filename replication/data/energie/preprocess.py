"""
Preprocess Austrian-German Pol.is comments about climate-related energy policy
(Klimarat-Energie) into policies.json.

Phase 0 (filter):
    - Drops moderated != '1'
    - Drops comments whose minor-group vote count is below MIN_GROUP_VOTES
      (default 10). For Klimarat-Energie this filter happens to remove nothing
      on top of moderation, but it keeps preprocessing robust if reused.

Phase 1 (translate):
    - Translates Austrian German -> English (gpt-5, thinking off), 10 workers
    - candidate_id = comment-id from the source CSV
    - Cached/resumable output: policies_full.json

Phase 2 (curate):
    - Drops URL/link spam and survey-meta entries (see DROP_IDS)
    - Sorts remaining by English text length descending (proxy for content depth)
    - Writes the curated set to policies.json

Run from project root:
    python data/energie/preprocess.py
"""
import csv
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from delta.inference import query


def _query_with_retry(query_input, inference_settings, max_retries=5, base_delay=20):
    """Wrap query() with exponential-backoff retry. Catches 429s + transient errors."""
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            return query(query_input, inference_settings, verbose=False)
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                msg = str(e)[:120]
                print(f"  query failed ({type(e).__name__}: {msg}); retrying in {delay}s...")
                time.sleep(delay)
    raise last_err

N_WORKERS = 10
MIN_GROUP_VOTES = 10

INPUT_CSV = "data/energie/comments.csv"
GROUPED_CSV = "data/energie/comments-grouped.csv"
POLICIES_FULL_JSON = "data/energie/policies_full.json"
POLICIES_JSON = "data/energie/policies.json"

# Non-policy / non-opinion entries — refine after reviewing translations.
DROP_IDS = set()

TRANSLATE_SETTINGS = {
    'model': 'gpt-5',
    'is_openrouter': False,
    'get_logprobs': False,
    'k_logprobs': 0,
    'thinking': False,
    'thinking_effort': "minimal",
    'temperature': 1,
    'max_output_tokens': 300,
}

SYSTEM_PROMPT = (
    "You are a careful translator. Translate Austrian-German opinion statements "
    "about climate-related energy policy (renewables, heating fuels, transport "
    "electrification, energy efficiency, grid, storage, fossil-fuel phase-out) "
    "into faithful, natural English. Preserve the speaker's hedging (e.g. "
    "render 'ich finde' as 'I think' or 'I feel' as appropriate), sentiment, "
    "and policy-specific terminology. Keep institutional names as-is (Klimarat, "
    "Klimakommission, EU, Bundesregierung, Bundesland, Gemeinde). When a "
    "comment is prefixed with 'Klimarat:' (a label marking it as a Klimarat "
    "proposition), preserve that prefix as 'Klimarat:'. Output ONLY the "
    "translated English sentence — no quotes, no commentary."
)


def load_cached_translations(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return {entry["original_text"]: entry["text"] for entry in json.load(f)}


def well_voted_ids(min_votes=MIN_GROUP_VOTES):
    """Comment-ids that have at least `min_votes` in both Pol.is groups."""
    g = pd.read_csv(GROUPED_CSV)
    keep = g[g[['group-a-votes', 'group-b-votes']].min(axis=1) >= min_votes]
    return set(keep['comment-id'].astype(str))


def translate_all():
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["moderated"] == "1"]
    print(f"Loaded {len(rows)} moderated=1 comments")

    well_voted = well_voted_ids()
    rows = [r for r in rows if r["comment-id"] in well_voted]
    print(f"After min(A,B) >= {MIN_GROUP_VOTES} vote filter: {len(rows)} comments")

    cached = load_cached_translations(POLICIES_FULL_JSON)
    if cached:
        print(f"Resuming: {len(cached)} translations cached")

    policies = [None] * len(rows)
    for i, row in enumerate(rows):
        de = row["comment-body"].strip()
        cid = row["comment-id"]
        if de in cached:
            policies[i] = {"candidate_id": cid, "text": cached[de], "original_text": de}

    todo = [(i, row) for i, row in enumerate(rows) if policies[i] is None]
    print(f"Translating {len(todo)} uncached rows with {N_WORKERS} workers")

    lock = threading.Lock()
    done = 0

    def save_locked():
        with open(POLICIES_FULL_JSON, "w", encoding="utf-8") as f:
            json.dump([p for p in policies if p is not None], f, ensure_ascii=False, indent=2)

    def translate_one(i, row):
        de = row["comment-body"].strip()
        cid = row["comment-id"]
        try:
            result = _query_with_retry(
                {"system_prompt": SYSTEM_PROMPT, "user_prompt": de},
                TRANSLATE_SETTINGS,
            )
        except Exception as e:
            print(f"  [ERR] giving up on comment-id={cid}: {type(e).__name__}: {str(e)[:120]}")
            result = None
        return i, cid, de, result

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = [ex.submit(translate_one, i, row) for i, row in todo]
        for fut in as_completed(futures):
            try:
                i, cid, de, result = fut.result()
            except Exception as e:
                print(f"  [ERR] worker raised after retries: {type(e).__name__}: {str(e)[:120]}")
                continue
            with lock:
                if result is None:
                    print(f"  [WARN] translation failed for comment-id={cid}: {de[:60]}")
                else:
                    en = result.strip().strip('"').strip("'")
                    policies[i] = {"candidate_id": cid, "text": en, "original_text": de}
                    done += 1
                    print(f"[{done}/{len(todo)}] {de[:40]} -> {en[:70]}")
                    if done % 20 == 0:
                        save_locked()

    with lock:
        save_locked()

    kept = [p for p in policies if p is not None]
    print(f"Wrote {len(kept)} translations to {POLICIES_FULL_JSON}")
    return kept


def curate(policies):
    kept = [p for p in policies if p["candidate_id"] not in DROP_IDS]
    print(f"Curate: dropped {len(policies) - len(kept)} non-policy entries -> {len(kept)} substantive")

    kept.sort(key=lambda p: len(p["text"]), reverse=True)
    print(f"Curate: sorted by length (longest={len(kept[0]['text'])} chars, shortest={len(kept[-1]['text'])} chars)")

    with open(POLICIES_JSON, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(kept)} substantive policies to {POLICIES_JSON}")


def main():
    policies = translate_all()
    curate(policies)


if __name__ == "__main__":
    main()
