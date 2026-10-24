# Browses the FIRE campus free-speech databases to shortlist candidate
# incidents for the campus-speech case set.
#
# This script does NOT produce ../cases.json. It narrows the databases to
# incidents whose description contains a direct quote, breaks them out by topic,
# and prints samples for reading. The final 20 cases were then selected and
# rewritten by hand from that shortlist. Requires the FIRE databases, which are
# not included here — see README.md in this directory.

# %% ------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# IMPORTS -------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #

import re
import textwrap
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# %% ------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# LOAD + MERGE DATA ---------------------------------------------------------- #
# ---------------------------------------------------------------------------- #

# The FIRE databases are not redistributed with this release (see README.md
# in this directory). Download them from FIRE and place the three CSVs here,
# or point DATA_DIR at wherever you keep them.
DATA_DIR = Path(__file__).resolve().parent / "fire-databases"

SOURCES = {
    "scholars":      "Scholars Under Fire Database.csv",
    "students":      "Students Under Fire Database.csv",
    "deplatforming": "Campus Deplatforming.csv",
}

frames = []
for source, fname in SOURCES.items():
    df = pd.read_csv(DATA_DIR / fname)
    df["source"] = source
    frames.append(df)

cases = pd.concat(frames, ignore_index=True)

print(f"Loaded {len(cases)} cases:")
print(cases["source"].value_counts())


# %% ------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# FILTER TO CASES WITH AN EXACT QUOTE ---------------------------------------- #
# ---------------------------------------------------------------------------- #

# Keep only cases whose controversy_explanation contains a paired set of
# double quotes (covering both straight "..." and curly "...").

QUOTE_PATTERN = re.compile(r'["\u201C].+?["\u201D]', flags=re.DOTALL)

has_quote = cases["controversy_explanation"].fillna("").str.contains(QUOTE_PATTERN)
cases_quoted = cases.loc[has_quote].reset_index(drop=True)

print(f"Cases with an exact quote in controversy_explanation: "
      f"{len(cases_quoted)} / {len(cases)}")


# %% ------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# EXPLODE TOPICS ------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #

# Each case may have multiple topics, separated by "; ". Split on "; " and
# explode so each (case, topic) pair gets its own row.

topics = (
    cases_quoted.assign(
        controversy_topics=cases["controversy_topics"]
            .fillna("")
            .str.split(r"\s*;\s*", regex=True)
    )
    .explode("controversy_topics")
    .assign(controversy_topics=lambda d: d["controversy_topics"].str.strip())
    .query("controversy_topics != ''")
    .reset_index(drop=True)
)


# %% ------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# PLOT HISTOGRAM ------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #

topic_counts = (
    topics.groupby(["controversy_topics", "source"])
    .size()
    .unstack(fill_value=0)
)

topic_counts = topic_counts.assign(_total=topic_counts.sum(axis=1)).sort_values(
    "_total", ascending=True
).drop(columns="_total")

source_order = ["scholars", "students", "deplatforming"]
topic_counts = topic_counts[[s for s in source_order if s in topic_counts.columns]]

fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(topic_counts))))
topic_counts.plot.barh(stacked=True, ax=ax, width=0.8)

for i, total in enumerate(topic_counts.sum(axis=1)):
    ax.text(total + max(topic_counts.sum(axis=1)) * 0.005, i, f"{int(total)}",
            va="center", fontsize=8)

ax.set_xlabel("Number of incidents")
ax.set_ylabel("")
ax.set_title("FIRE campus free speech incidents by topic\n(cases with an exact quote in controversy_explanation)")
ax.legend(title="Source", loc="lower right", frameon=False)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.show()


# %% ------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# SAMPLE INCIDENTS BY TOPIC -------------------------------------------------- #
# ---------------------------------------------------------------------------- #


def sample_incidents(topic, n=50, random_state=None, width=70):
    matches = topics[topics["controversy_topics"].str.casefold() == topic.casefold()]
    if matches.empty:
        available = sorted(topics["controversy_topics"].unique())
        raise ValueError(
            f"No incidents found for topic {topic!r}. "
            f"Available topics: {available}"
        )
    sample = matches.sample(n=min(n, len(matches)), random_state=random_state)

    for i, row in enumerate(sample.itertuples(index=False), start=1):
        header = f"[{i}] {row.name}  —  source: {row.source}"
        wrapped_header = textwrap.fill(header, width=width, subsequent_indent="    ")
        print(wrapped_header)
        print("-" * min(width, max(len(line) for line in wrapped_header.splitlines())))
        explanation = row.controversy_explanation if isinstance(row.controversy_explanation, str) else ""
        for paragraph in explanation.split("\n"):
            print(textwrap.fill(paragraph, width=width) if paragraph.strip() else "")
        print()

# %%

sample_incidents("Terrorism")

# %%