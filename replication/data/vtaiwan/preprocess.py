"""
Preprocess vTaiwan Uber consultation comments into policies.json.

Phase 1 (translate):
    - Filters to moderated == '1' (99 of 199 rows)
    - Translates Traditional Chinese -> English (gpt-5, thinking off)
    - candidate_id = comment-id from the source CSV
    - Cached/resumable output: policies_full.json (all 99 translations)

Phase 2 (curate):
    - Drops 5 factual self-disclosure entries (e.g. "I have a driver's license")
    - Drops 22 descriptive entries that compare Uber and taxis as they already are,
      rather than proposing what should be done
    - Sorts the remaining 72 by English text length descending (proxy for content depth)
    - Writes the 72 to policies.json — this is what delta-run evaluates against.
      Case-generation seed pool is controlled via scenario_params['n_top_policies_for_cases'].

Run from project root:
    python data/vtaiwan/preprocess.py
"""
import csv
import json
import os
from delta.inference import query

INPUT_CSV = "data/vtaiwan/comments.csv"
POLICIES_FULL_JSON = "data/vtaiwan/policies_full.json"
POLICIES_JSON = "data/vtaiwan/policies.json"

# Factual self-disclosure entries — not opinion statements, so they would not
# produce meaningful contrasts in the case-generation step. Keyed by comment-id.
DROP_IDS = {
    "0",   # I have used Uber to hail a ride.
    "10",  # I have used Uber outside Taiwan.
    "14",  # I have a driver's license for small cars.
    "15",  # I have a commercial driver's license.
    "43",  # I have taken an Uber with surge pricing.
}

# Descriptive comparisons of Uber and taxis as they currently operate. DELTA
# scores a statement by whether raters agree on which concrete actions it
# permits and rules out, so a claim about how things already are gives them
# nothing to apply to a case. Keyed by comment-id.
DESCRIPTIVE_IDS = {
    "38",  # If I’m not in a hurry, even if there are plenty of taxis by the ro...
    "39",  # I think UberX, on average, has better quality than taxis.
    "53",  # I think Uber is a new service that benefits both passengers and dr...
    "59",  # I think using online tools (such as apps) to hail a ride can reduc...
    "61",  # I think the sharing economy can reduce the waste of society’s reso...
    "66",  # I think Uber drivers are less likely to weave through traffic like...
    "72",  # I think UberX is usually cheaper than taxis in terms of fares, and...
    "84",  # I think traditional taxis now have to join a fleet to survive, and...
    "90",  # I think that when you file a complaint through the Uber app, issue...
    "161", # From a safety perspective, compared with a private-use license, th...
    "162", # As far as drivers are concerned, taxi drivers earn more than Uber ...
    "169", # I think Uber’s rating system is very good and effectively weeds ou...
    "170", # It used to be the cab companies that managed drivers; now it’s the...
    "171", # I think Uber’s tax avoidance is normal, just like how when you fil...
    "172", # I think the biggest difference between Uber and the taxi fleets wh...
    "121", # I think I take Uber because the drivers are generally better—unlik...
    "128", # I think that, when it comes to the cars, because they’re the drive...
    "135", # It’s often reported that in other counties and cities, white-plate...
    "137", # I think taking a taxi is a risky thing, because the quality of the...
    "67",  # With Uber, you might have the chance to ride in high-end models (e...
    "62",  # I think Uber is a business model that can create flexible employme...
    "159", # Setting aside the legal aspect, Uber has better matching efficienc...
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
    "You are a careful translator. Translate Traditional Chinese opinion statements "
    "about Uber/UberX ride-sharing policy in Taiwan into faithful, natural English. "
    "Preserve the speaker's hedging (e.g. render '我覺得' as 'I think' or 'I feel' as "
    "appropriate), sentiment, and policy-specific terminology. Keep 'UberX' and 'Uber' "
    "as-is. Output ONLY the translated English sentence — no quotes, no commentary."
)


def load_cached_translations(path):
    """Map original Chinese text -> existing English translation, for resuming."""
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

    policies = []
    for i, row in enumerate(rows):
        zh = row["comment-body"].strip()
        cid = row["comment-id"]

        if zh in cached:
            policies.append({"candidate_id": cid, "text": cached[zh], "original_text": zh})
            continue

        result = query(
            {"system_prompt": SYSTEM_PROMPT, "user_prompt": zh},
            TRANSLATE_SETTINGS,
            verbose=False,
        )
        if result is None:
            print(f"  [WARN] translation failed for comment-id={cid}: {zh}")
            continue
        en = result.strip().strip('"').strip("'")
        policies.append({"candidate_id": cid, "text": en, "original_text": zh})
        print(f"[{i+1}/{len(rows)}] {zh[:40]} -> {en[:70]}")

        with open(POLICIES_FULL_JSON, "w", encoding="utf-8") as f:
            json.dump(policies, f, ensure_ascii=False, indent=2)

    with open(POLICIES_FULL_JSON, "w", encoding="utf-8") as f:
        json.dump(policies, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(policies)} translations to {POLICIES_FULL_JSON}")
    return policies


def curate(policies):
    substantive = [p for p in policies if p["candidate_id"] not in DROP_IDS]
    print(f"Curate: dropped {len(policies) - len(substantive)} factual entries -> {len(substantive)} substantive")

    kept = [p for p in substantive if p["candidate_id"] not in DESCRIPTIVE_IDS]
    print(f"Curate: dropped {len(substantive) - len(kept)} descriptive entries -> {len(kept)} prescriptive")

    # Sort by English-text length descending (proxy for content depth). Index 0
    # is the longest, so generate_cases can seed from policies[:n] for any n.
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
