"""Build the judge-input CSV for the human-experiment (campus speech) policies.

Converts data/human_experiment/policies.json into the CSV schema that
rate_statements.py expects (question, candidate_id, iteration, round_id,
is_top_candidate, text). The university speech-code policies don't come from
a deliberation, so the boilerplate columns are synthesized: a single shared
question framing the domain, iteration=0, round_id="human_experiment",
is_top_candidate=True.

The adversarial dataset needs no builder; its input is committed at
analysis/adversarial/adversarial_policies.csv.
"""

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
POLICIES_JSON = HERE.parent.parent / "data" / "human_experiment" / "policies.json"
OUT_CSV = HERE / "human_experiment_input.csv"

QUESTION = "What expression should the university permit or prohibit on campus?"


def build(out_csv: Path = OUT_CSV) -> pd.DataFrame:
    policies = json.load(open(POLICIES_JSON))
    df = pd.DataFrame(policies)
    df["question"] = QUESTION
    df["iteration"] = 0
    df["round_id"] = "human_experiment"
    df["is_top_candidate"] = True
    df = df[["question", "candidate_id", "iteration", "round_id", "is_top_candidate", "text"]]
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    build()
