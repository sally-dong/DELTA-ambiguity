"""
Preprocess Austrian-German Pol.is comments about climate-related housing and
the built environment into policies.json.

Phase 1 (translate):
    - Filters to moderated == '1'
    - Translates Austrian German -> English (gpt-5, thinking off)
    - candidate_id = comment-id from the source CSV
    - Cached/resumable output: policies_full.json

Phase 2 (curate):
    - Drops URL/link spam and survey-meta entries (see DROP_IDS)
    - Sorts remaining by English text length descending (proxy for content depth)
    - Writes the curated set to policies.json — this is what run_experiment.py
      evaluates against. Case-generation seed pool is controlled via
      scenario_params['n_top_policies_for_cases'].

Run from project root:
    python data/wohnen/preprocess.py
"""
import csv
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from delta.inference import query

N_WORKERS = 10

INPUT_CSV = "data/wohnen/comments.csv"
POLICIES_FULL_JSON = "data/wohnen/policies_full.json"
POLICIES_JSON = "data/wohnen/policies.json"

# Non-policy / non-opinion entries — link spam, etc. Refine after reviewing
# translations. (Pure factual self-disclosure like "I am a tenant" / "I live
# in Eastern Austria" is already filtered by moderated=-1 upstream.)
DROP_IDS = {
    "464",  # "Mobility and living shouldn't be looked at separately!" + URL spam.
}

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
    "about climate-related housing and the built environment (heating, building "
    "renovation, spatial planning, vacancy, density, energy in dwellings) into "
    "faithful, natural English. Preserve the speaker's hedging (e.g. render "
    "'ich finde' as 'I think' or 'I feel' as appropriate), sentiment, and "
    "policy-specific terminology. Keep institutional names as-is (Klimarat, "
    "Klimakommission, EU, Bundesregierung, Bundesland, Gemeinde). When a "
    "comment is prefixed with 'Klimarat:' (a label marking it as a Klimarat "
    "proposition), preserve that prefix as 'Klimarat:'. Output ONLY the "
    "translated English sentence — no quotes, no commentary."
)


def load_cached_translations(path):
    """Map original German text -> existing English translation, for resuming."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return {entry["original_text"]: entry["text"] for entry in json.load(f)}


def translate_all():
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["moderated"] == "1"]
    print(f"Loaded {len(rows)} moderated=1 comments")

    cached = load_cached_translations(POLICIES_FULL_JSON)
    if cached:
        print(f"Resuming: {len(cached)} translations cached")

    # Pre-populate `policies` with cached entries at their input-order index, so
    # output order matches the source CSV regardless of completion order.
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
        result = query(
            {"system_prompt": SYSTEM_PROMPT, "user_prompt": de},
            TRANSLATE_SETTINGS,
            verbose=False,
        )
        return i, cid, de, result

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = [ex.submit(translate_one, i, row) for i, row in todo]
        for fut in as_completed(futures):
            i, cid, de, result = fut.result()
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
