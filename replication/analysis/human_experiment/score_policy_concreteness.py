"""Score the concreteness of each policy using the Brysbaert-style norms
in concreteness_norms_dictionary.csv.

By default, scores policies in data/human_experiment/policies.json. Pass
``--input <csv>`` to score the ``text`` column of a CSV (e.g. the adversarial
policy sets); ``candidate_id`` is preserved when present.

For each policy, emits:
  - total_words: token count after tokenization
  - matched: number of dictionary hits (each bigram or unigram = 1 hit)
  - score_sum: sum of Conc.M over hits
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DICT_PATH = HERE / "concreteness_norms_dictionary.csv"
DEFAULT_POLICIES_PATH = REPO_ROOT / "data" / "human_experiment" / "policies.json"
DEFAULT_OUTPUT_PATH = HERE / "policy_concreteness.csv"

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")


def load_dictionary(path):
    df = pd.read_csv(path)
    df["Word"] = df["Word"].astype(str).str.lower()
    uni_df = df[df["Bigram"] == 0]
    bi_df = df[df["Bigram"] == 1]
    unigrams = dict(zip(uni_df["Word"], uni_df["Conc.M"]))
    bigrams = dict(zip(bi_df["Word"], bi_df["Conc.M"]))
    return unigrams, bigrams


def tokenize(text):
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def score_text(tokens, unigrams, bigrams):
    """Greedy left-to-right: prefer bigram, fall back to unigram, else skip."""
    matched = 0
    score_sum = 0.0
    i = 0
    n = len(tokens)
    while i < n:
        if i + 1 < n:
            bg = f"{tokens[i]} {tokens[i + 1]}"
            if bg in bigrams:
                score_sum += bigrams[bg]
                matched += 1
                i += 2
                continue
        if tokens[i] in unigrams:
            score_sum += unigrams[tokens[i]]
            matched += 1
        i += 1
    return matched, score_sum


def load_policies(path):
    """Return a list of {candidate_id, text} dicts from a JSON or CSV path."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        with open(path) as f:
            policies = json.load(f)
        return [{"candidate_id": p["candidate_id"], "text": p["text"]} for p in policies]
    df = pd.read_csv(path)
    if "text" not in df.columns:
        raise ValueError(f"{path} has no 'text' column")
    if "candidate_id" not in df.columns:
        df = df.assign(candidate_id=range(len(df)))
    return df[["candidate_id", "text"]].to_dict(orient="records")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_POLICIES_PATH,
                        help="Path to policies.json or a CSV with a 'text' column")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output CSV path (default: alongside input as *_concreteness.csv)")
    args = parser.parse_args()

    input_path = args.input
    if args.output is not None:
        output_path = args.output
    elif input_path == DEFAULT_POLICIES_PATH:
        output_path = DEFAULT_OUTPUT_PATH
    else:
        output_path = input_path.with_name(f"{input_path.stem}_concreteness.csv")

    unigrams, bigrams = load_dictionary(DICT_PATH)
    policies = load_policies(input_path)

    rows = []
    for p in policies:
        tokens = tokenize(p["text"])
        matched, score_sum = score_text(tokens, unigrams, bigrams)
        rows.append({
            "candidate_id": p["candidate_id"],
            "total_words": len(tokens),
            "matched": matched,
            "score_sum": round(score_sum, 4),
        })

    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False)
    print(out.to_string(index=False))
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
