"""How the distribution of participant agreement shifts with ambiguity.

Produces the paper's Habermas figure: for each point on the 1-7 agreement
scale, the LOWESS-smoothed share of responses at that level as a function of
the winning statement's DELTA rank within its question, with 95% bands from
a cluster bootstrap over deliberation rounds.

Restricted to rounds whose participants were initially divided -- at least
one member above and one below neutral on the question -- since a round
that started in agreement cannot show ambiguity failing to bridge.

Reads:
    analysis/habermas/participant_agreements.csv
    analysis/habermas/participant_positions.csv
    data/q*/policies.json + results/q*/scores/delta_scores.csv

Writes:
    agreement_distribution_lowess.pdf

Usage:
    python analysis/habermas/habermas_agreement_distribution.py
"""
import argparse
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

sys.path.insert(0, os.path.dirname(HERE))
from plot_style import FIG_WIDTH, set_style

# Diverging red-blue, darkened so that all seven read as 1px lines. The
# light fills that work for stacked bars (slightly disagree / neutral /
# slightly agree) vanish against white as lines.
LINE_COLORS = {
    1: '#67001f', 2: '#b2182b', 3: '#d6604d',
    4: '#404040',
    5: '#4393c3', 6: '#2166ac', 7: '#053061',
}

# Positions strictly above / below the 4 = neutral midpoint. Participants at
# exactly 4 count toward neither side.
PRO_MIN, CON_MAX = 5, 3


def load_delta_scores():
    """candidate_id -> DELTA score, across the q1-q9 topic runs.

    Scores are stored by policy index, so they are joined back to statement
    IDs through each run's policies.json.
    """
    frames = []
    # q[0-9] only: the q6_trial2 / q9_trial2 re-runs score the same statements
    # again and would duplicate them into the panel.
    for scores_path in sorted(glob.glob(os.path.join(
            ROOT, 'results', 'q[0-9]', 'scores', 'delta_scores.csv'))):
        run = os.path.basename(os.path.dirname(os.path.dirname(scores_path)))
        policies_path = os.path.join(ROOT, 'data', run, 'policies.json')
        if not os.path.exists(policies_path):
            continue
        with open(policies_path, encoding='utf-8') as f:
            candidate_ids = [p['candidate_id'] for p in json.load(f)]
        scores = pd.read_csv(scores_path)
        scores['candidate_id'] = scores['policy'].map(dict(enumerate(candidate_ids)))
        frames.append(scores[['candidate_id', 'score']])

    if not frames:
        raise FileNotFoundError('no q[0-9] score files found under results/')
    scores = pd.concat(frames, ignore_index=True)
    if scores['candidate_id'].duplicated().any():
        raise ValueError('a statement is scored by more than one run; '
                         'the panel would double-count it')
    return scores.rename(columns={'score': 'delta_score'})


def build_panel(divided_only=True):
    """One row per (participant, round, iteration) with the participant's
    agreement and the winning statement's DELTA score."""
    agreements = pd.read_csv(os.path.join(HERE, 'participant_agreements.csv'))
    panel = agreements.merge(load_delta_scores(), on='candidate_id', how='left')

    if divided_only:
        positions = pd.read_csv(os.path.join(HERE, 'participant_positions.csv'))
        positions = positions.dropna(subset=['position_initial'])
        side = np.where(positions['position_initial'] >= PRO_MIN, 'pro',
                np.where(positions['position_initial'] <= CON_MAX, 'con', 'neutral'))
        counts = (positions.assign(side=side)
                  .groupby(['round_id', 'side']).size().unstack(fill_value=0)
                  .reindex(columns=['pro', 'con', 'neutral'], fill_value=0))
        divided = counts.index[(counts['pro'] >= 1) & (counts['con'] >= 1)]
        panel = panel[panel['round_id'].isin(divided)]

    return panel.dropna(subset=['agreement', 'delta_score']).copy()


def _cluster_bootstrap(x, y, groups, x_grid, frac, n_boot, rng):
    """95% band for a LOWESS fit, resampling whole rounds with replacement.

    Participants in a round share one winning statement and therefore one
    DELTA value, so their responses are correlated; resampling individual
    ratings would understate the uncertainty.
    """
    unique = np.unique(groups)
    index_by_round = {g: np.where(groups == g)[0] for g in unique}
    boots = np.full((n_boot, len(x_grid)), np.nan)
    for b in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index_by_round[g] for g in sampled])
        try:
            boots[b] = lowess(y[idx], x[idx], frac=frac, it=0, xvals=x_grid)
        except Exception:
            pass
    return np.nanpercentile(boots, 2.5, axis=0), np.nanpercentile(boots, 97.5, axis=0)


def _stagger(target_ys, min_gap=0.022, ymin=0.0, ymax=0.40):
    """Nudge endpoint labels apart where curves converge, preserving order."""
    order = np.argsort(target_ys)
    ys = np.array([target_ys[i] for i in order], dtype=float)
    for _ in range(50):
        moved = False
        for i in range(1, len(ys)):
            if ys[i] - ys[i - 1] < min_gap:
                nudge = (min_gap - (ys[i] - ys[i - 1])) / 2
                ys[i - 1] -= nudge
                ys[i] += nudge
                moved = True
        ys = np.clip(ys, ymin, ymax)
        if not moved:
            break
    out = np.zeros_like(ys)
    for k, i in enumerate(order):
        out[i] = ys[k]
    return out


def plot(panel, out_path, figsize=(FIG_WIDTH, 2.3), frac=0.45, n_boot=300, seed=42):
    set_style()

    sub = panel.copy()
    sub['agreement_int'] = sub['agreement'].round().astype(int).clip(1, 7)
    # Rank within question: DELTA scores are only comparable inside a run,
    # so the pooled x-axis has to be a per-question rank, not a raw score.
    sub['delta_rank'] = sub.groupby('question_short')['delta_score'].rank(pct=True)

    x = sub['delta_rank'].values
    groups = sub['round_id'].values
    x_grid = np.linspace(x.min(), 1.0, 80)
    rng = np.random.RandomState(seed)

    fig, ax = plt.subplots(figsize=figsize)
    endpoints = {}
    for cat in range(1, 8):
        y = (sub['agreement_int'] == cat).astype(float).values
        smooth = lowess(y, x, frac=frac, it=0, xvals=x_grid)
        lo, hi = _cluster_bootstrap(x, y, groups, x_grid, frac, n_boot, rng)
        ax.fill_between(x_grid, lo, hi, color=LINE_COLORS[cat], alpha=0.12, linewidth=0)
        ax.plot(x_grid, smooth, lw=1.2, color=LINE_COLORS[cat])
        endpoints[cat] = float(smooth[-1])

    placed = _stagger([endpoints[c] for c in range(1, 8)])
    for k, cat in enumerate(range(1, 8)):
        y0, y1 = endpoints[cat], placed[k]
        if abs(y1 - y0) > 0.003:
            ax.plot([1.0, 1.02], [y0, y1], color=LINE_COLORS[cat], lw=0.4, alpha=0.7)
        ax.text(1.025, y1, str(cat), fontsize=6.5, color=LINE_COLORS[cat],
                va='center', ha='left', fontweight='bold')

    ax.set_title('Agreement vs DELTA rank', pad=4)
    ax.set_xlabel('DELTA rank', labelpad=2)
    ax.set_ylabel('share of responses', labelpad=2)
    ax.set_xlim(0, 1.08)
    ax.set_ylim(0, 0.42)
    ax.set_yticks(np.arange(0, 0.41, 0.1))
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0', '.25', '.5', '.75', '1'])
    ax.set_yticks(np.arange(0, 0.41, 0.1))
    ax.tick_params(axis='both', length=3, pad=2)
    sns.despine(ax=ax)
    fig.tight_layout()
    # CreationDate=None keeps the PDF byte-identical across runs, so a
    # re-render does not show up as a spurious diff.
    fig.savefig(out_path, pad_inches=0.02, bbox_inches='tight',
                metadata={'CreationDate': None})
    plt.close(fig)
    return endpoints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all-rounds', action='store_true',
                        help='include rounds that were not initially divided')
    parser.add_argument('--out', default=os.path.join(HERE, 'agreement_distribution_lowess.pdf'))
    args = parser.parse_args()

    panel = build_panel(divided_only=not args.all_rounds)
    print(f'{len(panel)} responses, {panel["round_id"].nunique()} rounds, '
          f'{panel["question_short"].nunique()} questions')

    endpoints = plot(panel, args.out)
    print('share at the most ambiguous end:')
    for cat in range(1, 8):
        print(f'  {cat}: {endpoints[cat]*100:5.1f}%')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
