"""Build the Habermas assembly statement sets: data/q<n>/q<n>.csv and
policies.json.

Ported from the archived Habermas repository's extract_assembly_candidates.py,
which produced the committed q<n>.csv and policies.json files. The pipeline:

 1. Read hm_all_candidate_comparisons.parquet from the Habermas Machine
    release (Tessler et al. 2024) — the same download analysis/habermas/
    extract.py uses, from the same directory (see its NOTES.md).
 2. Identify the nine Virtual Citizens' Assembly questions (the ones with the
    most deliberation rounds) and, per round and iteration, recompute each
    candidate statement's mean rank from participant rankings. The release's
    top-candidate field is a placeholder; the winner is the candidate with
    the lowest mean rank, ties falling to the earlier candidate. Rounds
    missing either iteration are dropped.
 3. Apply the manually cleaned statement texts from cleaned_statements.csv
    (committed here): truncations of statements whose raw text contains
    misidentified content such as appended group commentary.
 4. Strip stray surrounding quotes, then deduplicate identical texts within a
    (round, iteration), merging their ratings.
 5. Partition by question into data/q<n>/q<n>.csv. Numbering follows the
    fixed topic order below; q8 (the Internet question) is excluded from the
    paper — it asks for a judgment of fact, not a prescription — and is not
    written.
 6. Write data/q<n>/policies.json: the is_top_candidate rows of q<n>.csv as
    [{candidate_id, text}] — the policy sets the DELTA runs score.

Steps 3-6 are verified to reproduce the committed files byte-for-byte from
the archived repository's intermediate CSV. Step 2 requires the ~400 MB
parquet download and re-runs the whole chain.

Usage:
    python data/extract_assembly_statements.py                # full pipeline
    python data/extract_assembly_statements.py --from-csv F   # skip parquet,
                                                              # start from an
                                                              # existing
                                                              # candidates CSV
    python data/extract_assembly_statements.py --policies-only  # step 6 only
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PARQUET = HERE.parent / "analysis" / "habermas" / "hm_all_candidate_comparisons.parquet"
CLEANED = HERE / "cleaned_statements.csv"
CANDIDATES_CSV = HERE / "assembly_questions_candidates.csv"  # intermediate, untracked

# Fixed question numbering (q1..q9). The paper's datasets are q1-q7 and q9;
# q8 (Internet) is not a prescriptive question and is skipped.
TOPIC_ORDER = [
    "prison", "minimum wage", "retirement age", "asylum",
    "european union", "national pride", "childcare",
    "internet", "carbon neutral",
]
SKIP_QUESTIONS = {8}

AGREEMENT_MAP = {
    "STRONGLY_DISAGREE": 1, "DISAGREE": 2, "SOMEWHAT_DISAGREE": 3,
    "NEUTRAL": 4, "SOMEWHAT_AGREE": 5, "AGREE": 6, "STRONGLY_AGREE": 7,
}
QUALITY_MAP = {
    "TERRIBLE_QUALITY": 1, "POOR_QUALITY": 2, "SOMEWHAT_POOR_QUALITY": 3,
    "NEUTRAL": 4, "SOMEWHAT_GOOD_QUALITY": 5, "GOOD_QUALITY": 6,
    "EXCELLENT_QUALITY": 7,
}


###############################################################
########## Step 2: parquet -> candidates CSV ##################
###############################################################

def identify_assembly_questions(df):
    """The nine assembly questions are the ones with the most rounds."""
    round_counts = df.groupby("question.text")["round_id"].nunique().sort_values(ascending=False)
    questions = round_counts.head(9).index.tolist()
    for q in questions:
        print(f"  - {q[:60]}... ({round_counts[q]} rounds)")
    return questions


def compute_candidate_stats(df_iteration):
    """Per-candidate rank/agreement/quality lists for one round+iteration."""
    candidate_ids = list(df_iteration.iloc[0]["candidates.metadata.id"])
    candidate_texts = list(df_iteration.iloc[0]["candidates.text"])

    stats = {cid: {"ranks": [], "agreements": [], "qualities": []} for cid in candidate_ids}

    for _, row in df_iteration.iterrows():
        ranking_cids = list(row["rankings.candidate_ids"])
        ranks = list(row["rankings.numerical_ranks"])
        agreements = list(row["ratings.agreement"])
        qualities = list(row["ratings.quality"])

        for i, cid in enumerate(ranking_cids):
            if cid in stats:
                if ranks[i] != -1:  # -1 encodes a missing ranking
                    stats[cid]["ranks"].append(int(ranks[i]))
                if i < len(agreements) and agreements[i] in AGREEMENT_MAP:
                    stats[cid]["agreements"].append(AGREEMENT_MAP[agreements[i]])
                if i < len(qualities) and qualities[i] in QUALITY_MAP:
                    stats[cid]["qualities"].append(QUALITY_MAP[qualities[i]])

    result = {}
    for i, cid in enumerate(candidate_ids):
        s = stats[cid]
        result[cid] = {
            "candidate_id": cid,
            "text": candidate_texts[i] if i < len(candidate_texts) else None,
            "n_raters": len(s["ranks"]),
            "ranks": s["ranks"],
            "agreements": s["agreements"],
            "qualities": s["qualities"],
            "_avg_rank": np.mean(s["ranks"]) if s["ranks"] else None,
        }
    return result


def extract_candidates(df, assembly_questions):
    """One row per candidate per (question, round, iteration), winners
    recomputed from participant rankings. Rounds missing an iteration are
    dropped."""
    rows = []
    df_assembly = df[df["question.text"].isin(assembly_questions)]

    for question in assembly_questions:
        df_q = df_assembly[df_assembly["question.text"] == question]
        topic_id = int(df_q.iloc[0]["question.topic"])
        n_complete = 0

        for round_id in df_q["round_id"].unique():
            df_round = df_q[df_q["round_id"] == round_id]
            n_participants = len(df_round[df_round["iteration_index"] == 0])

            per_iteration = {}
            for iteration in [0, 1]:
                df_iter = df_round[df_round["iteration_index"] == iteration]
                if len(df_iter) == 0:
                    continue

                candidate_stats = compute_candidate_stats(df_iter)
                valid = [
                    c for c in candidate_stats.values()
                    if c["text"] is not None and str(c["text"]).strip() and c["n_raters"] > 0
                ]
                # Sorted stably by mean rank, so ties fall to the earlier
                # candidate; the winner is the first entry.
                ordered = sorted(
                    valid,
                    key=lambda x: x["_avg_rank"] if x["_avg_rank"] is not None else float("inf"),
                )
                top_id = ordered[0]["candidate_id"] if ordered else None
                per_iteration[iteration] = (ordered, top_id)

            if len(per_iteration) != 2:
                continue  # incomplete deliberation
            n_complete += 1

            for iteration, (ordered, top_id) in per_iteration.items():
                for c in ordered:
                    rows.append({
                        "question": question,
                        "topic": topic_id,
                        "round_id": round_id,
                        "iteration": iteration,
                        "n_participants": n_participants,
                        "candidate_id": c["candidate_id"],
                        "text": str(c["text"]),
                        "n_raters": c["n_raters"],
                        "ranks": json.dumps(c["ranks"]),
                        "avg_rank": round(np.mean(c["ranks"]), 3) if c["ranks"] else None,
                        "agreements": json.dumps(c["agreements"]),
                        "avg_agreement": round(np.mean(c["agreements"]), 3) if c["agreements"] else None,
                        "qualities": json.dumps(c["qualities"]),
                        "avg_quality": round(np.mean(c["qualities"]), 3) if c["qualities"] else None,
                        "is_top_candidate": c["candidate_id"] == top_id,
                    })

        print(f"  Processed: {question[:50]}... ({n_complete} complete rounds)")

    fieldnames = ["question", "topic", "round_id", "iteration", "n_participants",
                  "candidate_id", "text", "n_raters", "ranks", "avg_rank",
                  "agreements", "avg_agreement", "qualities", "avg_quality",
                  "is_top_candidate"]
    with open(CANDIDATES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {len(rows)} rows to {CANDIDATES_CSV}")


###############################################################
########## Steps 3-4: clean and deduplicate ###################
###############################################################

def apply_cleaned_statements(csv_path=CANDIDATES_CSV, cleaned_path=CLEANED):
    """Replace statement texts with the manually cleaned versions."""
    df = pd.read_csv(csv_path)
    cleaned_df = pd.read_csv(cleaned_path)
    cleaned_text_map = dict(zip(cleaned_df["candidate_id"], cleaned_df["text"]))

    changes = 0

    def get_cleaned_text(row):
        nonlocal changes
        cleaned = cleaned_text_map.get(row["candidate_id"])
        if cleaned is not None and row["text"] != cleaned:
            changes += 1
            return cleaned
        return row["text"]

    df["text"] = df.apply(get_cleaned_text, axis=1)
    df.to_csv(csv_path, index=False)
    print(f"  Replaced text for {changes} statements")


def clean_and_deduplicate(csv_path=CANDIDATES_CSV):
    """Strip stray surrounding quotes, drop the rank columns, and merge
    duplicate texts within a (round, iteration), pooling their ratings."""
    df = pd.read_csv(csv_path)
    original_count = len(df)

    def clean_quotes(text):
        if pd.isna(text):
            return text
        text = str(text).strip()
        quote_chars = "\"'“”‘’"
        while len(text) >= 2 and text[0] in quote_chars and text[-1] in quote_chars:
            text = text[1:-1].strip()
        return text

    df["text"] = df["text"].apply(clean_quotes)
    df = df.drop(columns=[c for c in ["ranks", "avg_rank"] if c in df.columns])

    df["agreements_list"] = df["agreements"].apply(lambda x: json.loads(x) if pd.notna(x) else [])
    df["qualities_list"] = df["qualities"].apply(lambda x: json.loads(x) if pd.notna(x) else [])

    def aggregate_duplicates(group):
        if len(group) == 1:
            return group.iloc[0]
        result = group.iloc[0].copy()
        all_agreements, all_qualities, total_raters = [], [], 0
        for _, row in group.iterrows():
            all_agreements.extend(row["agreements_list"])
            all_qualities.extend(row["qualities_list"])
            total_raters += row["n_raters"]
        result["n_raters"] = total_raters
        result["agreements"] = json.dumps(all_agreements)
        result["qualities"] = json.dumps(all_qualities)
        result["avg_agreement"] = round(np.mean(all_agreements), 3) if all_agreements else None
        result["avg_quality"] = round(np.mean(all_qualities), 3) if all_qualities else None
        result["is_top_candidate"] = any(row["is_top_candidate"] for _, row in group.iterrows())
        return result

    deduplicated_rows = [
        aggregate_duplicates(group)
        for _, group in df.groupby(["text", "round_id", "iteration"], sort=False)
    ]
    df_dedup = pd.DataFrame(deduplicated_rows)
    df_dedup = df_dedup.drop(columns=["agreements_list", "qualities_list"])

    column_order = ["question", "topic", "round_id", "iteration", "n_participants",
                    "candidate_id", "text", "n_raters", "agreements", "avg_agreement",
                    "qualities", "avg_quality", "is_top_candidate"]
    df_dedup = df_dedup[[c for c in column_order if c in df_dedup.columns]]
    df_dedup.to_csv(csv_path, index=False)
    print(f"  Deduplicated {original_count} -> {len(df_dedup)} rows")


###############################################################
########## Steps 5-6: partition and policy sets ###############
###############################################################

def question_number(question_text):
    """Position of the question in the fixed q1..q9 numbering."""
    q = question_text.lower()
    for i, key in enumerate(TOPIC_ORDER, start=1):
        if key in q:
            return i
    raise ValueError(f"question matches no known topic: {question_text[:60]}")


def partition_by_question(csv_path=CANDIDATES_CSV):
    """Write data/q<n>/q<n>.csv for every question the paper uses."""
    df = pd.read_csv(csv_path)
    for q_text in df["question"].unique():
        n = question_number(q_text)
        if n in SKIP_QUESTIONS:
            print(f"  q{n}: skipped (not a prescriptive question)")
            continue
        out_dir = HERE / f"q{n}"
        os.makedirs(out_dir, exist_ok=True)
        df_q = df[df["question"] == q_text]
        out_path = out_dir / f"q{n}.csv"
        df_q.to_csv(out_path, index=False)
        print(f"  q{n}: {len(df_q)} rows -> {out_path}")


def write_policies():
    """Write each question's policies.json: the winning (is_top_candidate)
    statements of q<n>.csv, in file order, as [{candidate_id, text}]."""
    for n in range(1, 10):
        if n in SKIP_QUESTIONS:
            continue
        q_csv = HERE / f"q{n}" / f"q{n}.csv"
        df = pd.read_csv(q_csv)
        top = df[df["is_top_candidate"]]
        policies = [
            {"candidate_id": row["candidate_id"], "text": row["text"]}
            for _, row in top.iterrows()
        ]
        out_path = HERE / f"q{n}" / "policies.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(policies, f, ensure_ascii=False, indent=2)
        print(f"  q{n}: {len(policies)} policies -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from-csv", type=str, default=None,
                        help="Skip the parquet stage; copy this candidates CSV "
                             "in as the intermediate and run cleaning onward")
    parser.add_argument("--policies-only", action="store_true",
                        help="Only rebuild policies.json from the existing q<n>.csv files")
    args = parser.parse_args()

    if args.policies_only:
        write_policies()
        return

    if args.from_csv:
        pd.read_csv(args.from_csv).to_csv(CANDIDATES_CSV, index=False)
        print(f"Starting from {args.from_csv}")
    else:
        print(f"Loading {PARQUET}...")
        df = pd.read_parquet(PARQUET)
        print(f"  {len(df)} rows")
        questions = identify_assembly_questions(df)
        extract_candidates(df, questions)

    apply_cleaned_statements()
    clean_and_deduplicate()
    partition_by_question()
    write_policies()


if __name__ == "__main__":
    main()
