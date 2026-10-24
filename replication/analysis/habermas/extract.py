"""Extract participant agreement and position ratings for the Habermas
Virtual Citizens' Assembly.

Source: the Habermas Machine release (Tessler et al. 2024),
https://github.com/google-deepmind/habermas_machine. Download

    hm_all_candidate_comparisons.parquet     (~400 MB)
    hm_all_position_statement_ratings.parquet

into this directory. They are gitignored; the two CSVs written here are
small and are tracked.

In the raw data a participant's judgments live in parallel arrays on one
row -- `rankings.candidate_ids` indexes into `rankings.numerical_ranks`,
`ratings.agreement` and `ratings.quality`. This unnests them to one row
per (participant, round, iteration, candidate).

Writes:
    participant_agreements.csv   one row per participant per winning
                                 statement: agreement and quality on the
                                 1-7 scale, plus that participant's rank.
    participant_positions.csv    one row per participant per round:
                                 pre- and post-deliberation agreement
                                 with the question's affirming statement.

Usage:
    python analysis/habermas/extract.py
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# The eight assembly questions used in the paper. The ninth question in the
# release ("Has the Internet been a net positive for humanity?") is dropped:
# it is not a prescriptive question, so it has no policy to be ambiguous.
QUESTION_LABELS = {
    "Should we be trying to reduce the number of people in prison?": "prison",
    "Should we raise the minimum wage in the UK?": "minimum_wage",
    "Should we raise the retirement age from 66 to 68?": "retirement_age",
    "Should we be doing more to encourage a greater degree of national pride?": "national_pride",
    "Should the UK change its asylum seeking policy to make it easier to enter the country?": "asylum_policy",
    "Should the UK apply to rejoin the European Union?": "rejoin_eu",
    "Should the government provide universal free childcare from birth?": "free_childcare",
    "Should the UK set a goal to be carbon neutral (reach net zero) by 2050?": "carbon_neutral",
}

# Categorical rating labels -> 1-7. MOCK (the control condition) is absent
# from both maps and so becomes NaN, which is what we want.
AGREEMENT_MAP = {
    'STRONGLY_DISAGREE': 1, 'DISAGREE': 2, 'SOMEWHAT_DISAGREE': 3,
    'NEUTRAL': 4,
    'SOMEWHAT_AGREE': 5, 'AGREE': 6, 'STRONGLY_AGREE': 7,
}
QUALITY_MAP = {
    'TERRIBLE_QUALITY': 1, 'POOR_QUALITY': 2, 'SOMEWHAT_POOR_QUALITY': 3,
    'NEUTRAL': 4,
    'SOMEWHAT_GOOD_QUALITY': 5, 'GOOD_QUALITY': 6, 'EXCELLENT_QUALITY': 7,
}

MISSING_RANK = -1  # the release's sentinel for "participant gave no ranking"


def load_comparisons():
    """Candidate comparisons for the eight assembly questions, restricted to
    rounds that completed both iterations."""
    df = pd.read_parquet(
        os.path.join(HERE, 'hm_all_candidate_comparisons.parquet'),
        columns=['round_id', 'iteration_index', 'question.text', 'question.id',
                 'own_opinion.metadata.participant_id',
                 'candidates.metadata.id', 'candidates.text',
                 'rankings.candidate_ids', 'rankings.numerical_ranks',
                 'ratings.agreement', 'ratings.quality'])
    df = df[df['question.text'].isin(QUESTION_LABELS)].copy()
    df['question_short'] = df['question.text'].map(QUESTION_LABELS)

    iters = df.groupby('round_id')['iteration_index'].agg(set)
    complete = iters.index[iters.apply(lambda s: {0, 1} <= s)]
    return df[df['round_id'].isin(complete)].copy()


def find_winning_statements(df):
    """The statement each round settled on, per iteration.

    The release's `top_candidate.metadata.id` is a dummy placeholder, so the
    winner has to be recomputed as the candidate with the lowest mean rank
    across the round's participants. Candidates with no text or no rankings
    are ineligible. Ties fall to the earlier candidate, matching the order
    the participants saw.
    """
    winners = []
    for (round_id, iteration), group in df.groupby(['round_id', 'iteration_index'],
                                                    sort=False):
        # Every participant in a round is shown the same candidates in the
        # same order, so the first row's list stands for the round.
        candidate_ids = list(group.iloc[0]['candidates.metadata.id'])
        texts = list(group.iloc[0]['candidates.text'])
        ranks = {cid: [] for cid in candidate_ids}

        for _, row in group.iterrows():
            for i, cid in enumerate(row['rankings.candidate_ids']):
                if cid in ranks and row['rankings.numerical_ranks'][i] != MISSING_RANK:
                    ranks[cid].append(int(row['rankings.numerical_ranks'][i]))

        eligible = [(cid, ranks[cid]) for i, cid in enumerate(candidate_ids)
                    if i < len(texts) and texts[i] and str(texts[i]).strip()
                    and ranks[cid]]
        if not eligible:
            continue
        winner = min(eligible, key=lambda pair: np.mean(pair[1]))[0]
        winners.append((round_id, iteration, winner))

    return pd.DataFrame(winners, columns=['round_id', 'iteration', 'candidate_id'])


def extract_agreements(df, winners):
    """One row per participant per winning statement."""
    rows = []
    for row in df.itertuples(index=False):
        (round_id, iteration, _question, question_id, participant_id,
         _cids, _texts, ranking_ids, numerical_ranks, agreements, qualities,
         question_short) = row
        for i, cid in enumerate(ranking_ids):
            rank = numerical_ranks[i] if i < len(numerical_ranks) else MISSING_RANK
            rows.append((
                participant_id, round_id, question_id, question_short, iteration, cid,
                AGREEMENT_MAP.get(agreements[i]) if i < len(agreements) else None,
                QUALITY_MAP.get(qualities[i]) if i < len(qualities) else None,
                int(rank) if rank != MISSING_RANK else None,
            ))

    long = pd.DataFrame(rows, columns=[
        'participant_id', 'round_id', 'question_id', 'question_short',
        'iteration', 'candidate_id', 'agreement', 'quality', 'rank'])

    out = long.merge(winners, on=['round_id', 'iteration', 'candidate_id'])
    return out.sort_values(['question_short', 'round_id', 'iteration',
                            'participant_id']).reset_index(drop=True)


def extract_positions(df):
    """Pre- and post-deliberation position ratings, one row per participant
    per round.

    The position-ratings file carries no `round_id`, so rounds are attached
    through (participant, question) from the comparisons table.
    """
    psr = pd.read_parquet(
        os.path.join(HERE, 'hm_all_position_statement_ratings.parquet'),
        columns=['question.text', 'question.id', 'rating_index',
                 'ratings.agreement', 'metadata.participant_id'])
    psr = psr[psr['question.text'].isin(QUESTION_LABELS)].copy()
    psr['agreement'] = psr['ratings.agreement'].apply(
        lambda a: AGREEMENT_MAP.get(a[0]) if len(a) else None)

    wide = (psr.pivot_table(index=['metadata.participant_id', 'question.id'],
                            columns='rating_index', values='agreement',
                            aggfunc='first')
              .rename(columns={0: 'position_initial', 1: 'position_final'})
              .reset_index()
              .rename(columns={'metadata.participant_id': 'participant_id',
                               'question.id': 'question_id'}))

    rounds = (df[['own_opinion.metadata.participant_id', 'question.id',
                  'round_id', 'question_short']]
              .rename(columns={'own_opinion.metadata.participant_id': 'participant_id',
                               'question.id': 'question_id'})
              .drop_duplicates())

    out = rounds.merge(wide, on=['participant_id', 'question_id'], how='left')
    for col in ('position_initial', 'position_final'):
        if col not in out:
            out[col] = np.nan
    out = out[['participant_id', 'round_id', 'question_id', 'question_short',
               'position_initial', 'position_final']]
    return out.sort_values(['question_short', 'round_id',
                            'participant_id']).reset_index(drop=True)


def main():
    df = load_comparisons()
    print(f'{df["round_id"].nunique()} complete rounds over '
          f'{df["question_short"].nunique()} questions')

    winners = find_winning_statements(df)
    print(f'{len(winners)} winning statements')

    agreements = extract_agreements(df, winners)
    path = os.path.join(HERE, 'participant_agreements.csv')
    agreements.to_csv(path, index=False)
    print(f'wrote {path}  ({len(agreements)} rows, '
          f'{agreements["participant_id"].nunique()} participants)')

    positions = extract_positions(df)
    path = os.path.join(HERE, 'participant_positions.csv')
    positions.to_csv(path, index=False)
    print(f'wrote {path}  ({len(positions)} rows)')


if __name__ == '__main__':
    main()
