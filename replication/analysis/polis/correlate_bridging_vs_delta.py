"""Correlate Pol.is group-aware-consensus ("bridging") with the DELTA
ambiguity score, per dataset.

Bridging per comment (Pol.is convention, Laplace-smoothed):

    p_a = (agrees_A + 1) / (agrees_A + disagrees_A + 2)
    p_b = (agrees_B + 1) / (agrees_B + disagrees_B + 2)
    gac_agree    = p_a * p_b              # both groups tend to agree
    gac_disagree = (1 - p_a) * (1 - p_b)  # both groups tend to disagree
    bridging     = max(gac_agree, gac_disagree)  # whichever direction unites them
    divergence   = |p_a - p_b|

A high `bridging` value means both groups land on the same side of the comment
(broad consensus); a high `divergence` value means the groups split.

Per dataset, joins to `delta_scores.csv` via the policy index ->
candidate_id (= comment-id) mapping in `policies.json`, then reports Pearson
and Spearman correlations.

Usage:
    python analysis/polis/correlate_bridging_vs_delta.py --dir produktion
    python analysis/polis/correlate_bridging_vs_delta.py --dir all
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_style import FIG_WIDTH, set_style

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

# Every dataset keeps its per-group vote table at data/{dir}/comments-grouped.csv.
GROUPED_CSV = 'comments-grouped.csv'

# dataset name -> default results dir
DATASETS = {
    'vtaiwan':    {'results_dir': 'results/vtaiwan'},
    'produktion': {'results_dir': 'results/produktion'},
    'wohnen':     {'results_dir': 'results/wohnen'},
    'energie':    {'results_dir': 'results/energie'},
}

# Display names for paper figures
DATASET_DISPLAY = {
    'vtaiwan':    'Uber',
    'produktion': 'Climate - consumption',
    'wohnen':     'Climate - housing',
    'energie':    'Climate - energy',
}

METRIC_DISPLAY = {
    'bridging':             'bridging (GAC)',
    'bridging_min':         'bridging',
    'polarization_gac':     'polarization',
    'in_group_support_gac': 'in-group support (union)',
    'divergence':           'divergence',
    'in_group_support':     'in-group support',
    'gac_agree':            'GAC agree',
    'gac_disagree':         'GAC disagree',
    'p_a_agree':            'group A agreement',
    'p_b_agree':            'group B agreement',
    'upvote_rate':          'upvote rate',
}


def compute_bridging(grouped_csv):
    df = pd.read_csv(grouped_csv)
    a_a, a_d = df['group-a-agrees'], df['group-a-disagrees']
    b_a, b_d = df['group-b-agrees'], df['group-b-disagrees']
    p_a = (a_a + 1) / (a_a + a_d + 2)
    p_b = (b_a + 1) / (b_a + b_d + 2)
    gac_agree    = p_a * p_b
    gac_disagree = (1 - p_a) * (1 - p_b)
    # Overall upvote rate over all voters (both groups + ungrouped), Laplace-smoothed.
    # Treats passes as not a vote, mirroring the per-group p_a / p_b formula.
    tot_a, tot_d = df['total-agrees'], df['total-disagrees']
    upvote_rate = (tot_a + 1) / (tot_a + tot_d + 2)
    # Fractional reach within each cluster: per-comment vote count divided by
    # estimated cluster size (= max votes that cluster cast on any single
    # comment). Lets us filter on "fraction of group that engaged" rather than
    # absolute vote counts, which is more meaningful when clusters differ in
    # size by orders of magnitude across datasets.
    group_a_size = max(df['group-a-votes'].max(), 1)
    group_b_size = max(df['group-b-votes'].max(), 1)
    frac_a_seen = df['group-a-votes'] / group_a_size
    frac_b_seen = df['group-b-votes'] / group_b_size
    out = pd.DataFrame({
        'comment-id': df['comment-id'].astype(int),
        'comment':    df['comment'],
        'p_a_agree':    p_a,
        'p_b_agree':    p_b,
        'gac_agree':    gac_agree,
        'gac_disagree': gac_disagree,
        'bridging':     np.maximum(gac_agree, gac_disagree),
        # True bridging: both groups land high on agreement. Unlike `bridging`
        # above (Pol.is GAC), this does NOT reward "both groups disagree" — a
        # shared rejection isn't bringing people together.
        'bridging_min': np.minimum(p_a, p_b),
        'divergence':   (p_a - p_b).abs(),
        'upvote_rate':  upvote_rate,
        # In-group support: max one-sided agreement. High value = at least one
        # group strongly endorses the comment (whether or not the other does).
        # Complement of the min(p_a, p_b) framing of bridging.
        'in_group_support': np.maximum(p_a, p_b),
        # Multiplicative analog of in_group_support, in the Pol.is GAC family:
        # = P(at least one group endorses) under independence
        # = 1 - (1 - p_a)(1 - p_b)
        # = 1 - gac_disagree  (so this + gac_disagree partition [0,1])
        'in_group_support_gac': 1 - (1 - p_a) * (1 - p_b),
        # Multiplicative analog of divergence ("Pol.is divisive" / polarization):
        # = max(P(A agrees, B disagrees), P(A disagrees, B agrees)) under indep.
        # Hits its max at (p_a, p_b) = (1, 0) or (0, 1) — clean partition partner
        # to bridging, which hits its max at (1, 1) or (0, 0).
        'polarization_gac': np.maximum(p_a * (1 - p_b), (1 - p_a) * p_b),
        'frac_a_seen': frac_a_seen,
        'frac_b_seen': frac_b_seen,
        'frac_min_seen': np.minimum(frac_a_seen, frac_b_seen),
    })
    return out


def load_merged(dataset):
    """Load the per-policy merged frame for one dataset: delta score + bridging
    + per-group vote counts (so callers can filter by min-votes).
    """
    cfg = DATASETS[dataset]
    data_dir = ROOT / 'data' / dataset
    results_dir = ROOT / cfg['results_dir']

    bridging = compute_bridging(data_dir / GROUPED_CSV)

    delta = pd.read_csv(results_dir / 'scores' / 'delta_scores.csv')
    with open(data_dir / 'policies.json') as f:
        policies = json.load(f)
    cid_map = {i: int(p['candidate_id']) for i, p in enumerate(policies)}
    delta['comment-id'] = delta['policy'].map(cid_map).astype(int)

    grouped = pd.read_csv(data_dir / GROUPED_CSV)[
        ['comment-id', 'group-a-votes', 'group-b-votes']
    ]
    return delta.merge(bridging, on='comment-id', how='inner').merge(grouped, on='comment-id')


def _filter_by_frac(merged, min_frac):
    """Keep comments where each group's vote-reach fraction is >= min_frac."""
    if min_frac <= 0:
        return merged
    return merged[merged['frac_min_seen'] >= min_frac]


def _filter_by_votes(merged, min_votes):
    if min_votes <= 0:
        return merged
    keep = merged[['group-a-votes', 'group-b-votes']].min(axis=1) >= min_votes
    return merged[keep]


def analyse(dataset, min_votes=0):
    merged = load_merged(dataset)
    filtered = _filter_by_votes(merged, min_votes)

    print(f"\n=== {dataset} (min_votes >= {min_votes}) ===")
    print(f"  {len(filtered)}/{len(merged)} policies kept after vote filter")
    cols = ['bridging', 'gac_agree', 'gac_disagree', 'divergence',
            'p_a_agree', 'p_b_agree']
    print(f"  {'metric':14s}  {'Pearson r':>10s}   {'Spearman ρ':>11s}")
    score_rank = filtered['score'].rank()
    for col in cols:
        r   = filtered['score'].corr(filtered[col], method='pearson')
        rho = score_rank.corr(filtered[col].rank(), method='pearson')
        print(f"  {col:14s}  {r:+10.3f}   {rho:+11.3f}")

    out = HERE / f'bridging_vs_delta_{dataset}.csv'
    merged.to_csv(out, index=False)
    print(f"  -> wrote unfiltered merged table {out}")
    return filtered


def _spearman(merged, metric):
    return merged['score'].rank().corr(merged[metric].rank())


def _corr_with_ci(merged, metric='bridging', method='spearman'):
    """Correlation (Spearman ρ or Pearson r) plus a Fisher-z 95% CI. n_eff
    is taken as the sample count; the CI widens as filtering shrinks n.
    """
    if method == 'pearson':
        r = merged['score'].corr(merged[metric], method='pearson')
    else:
        r = _spearman(merged, metric)
    n = len(merged)
    if n <= 3 or abs(r) >= 1:
        return r, np.nan, np.nan
    # Fisher z = arctanh(r); SE_z ≈ 1/sqrt(n-3). For Spearman the (1+r²/2)
    # correction (Bonett & Wright) is small at moderate n; plain form stays
    # meaningful when |r| is large.
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    crit = 1.959963984540054  # two-sided 95% z; avoids the scipy dep
    lo = np.tanh(z - crit * se)
    hi = np.tanh(z + crit * se)
    return r, lo, hi


def panel_plot(datasets, metric, min_frac, out_path, use_ranks=False):
    """2x2 scatter panel: one panel per dataset, DELTA score (x) vs metric (y),
    filtered to comments with `min_frac` reach in both groups.

    If use_ranks=True, both axes are rank-transformed within each panel.
    """
    set_style()
    # Both variants force square panels (set_box_aspect below) so the figure
    # shape is identical across the two PDFs. hspace=0 keeps the inter-row
    # gap minimal; any vertical slack from the square-aspect constraint goes
    # into the outer top/bottom margins rather than between rows.
    fig, axes = plt.subplots(
        2, 2, figsize=(FIG_WIDTH, 3.4),
        sharex=False, sharey=not use_ranks,
        layout="constrained",
        gridspec_kw={'wspace': 0.25, 'hspace': 0.0},
    )
    axes_flat = axes.flatten()

    metric_label = METRIC_DISPLAY.get(metric, metric)
    x_label = "DELTA rank" if use_ranks else "DELTA score"
    y_label = (f"{metric_label.capitalize()} rank" if use_ranks
               else metric_label.capitalize())

    for ax, name in zip(axes_flat, datasets):
        merged = load_merged(name)
        f = _filter_by_frac(merged, min_frac).reset_index(drop=True)
        x_raw, y_raw = f['score'], f[metric]

        if use_ranks:
            res = stats.spearmanr(x_raw, y_raw)
            corr, pval, corr_label = res.correlation, res.pvalue, "ρ"
            x, y = x_raw.rank(), y_raw.rank()
        else:
            res = stats.pearsonr(x_raw, y_raw)
            corr, pval, corr_label = res.statistic, res.pvalue, "r"
            x, y = x_raw, y_raw
        p_text = "p < 0.001" if pval < 0.001 else f"p = {pval:.3f}"

        if use_ranks:
            # y=x reference diagonal — light gray dashed, behind data points.
            n = len(f)
            ax.plot([1, n], [1, n], color='lightgray', lw=0.7,
                    dashes=(4, 3), zorder=0)

        ax.scatter(x, y, s=8, color='#1f77b4', edgecolor='none', alpha=0.75)

        if not use_ranks:
            # OLS regression line on raw scores; suppressed for the rank version.
            slope, intercept = np.polyfit(x, y, 1)
            xx = np.array([x.min(), x.max()])
            ax.plot(xx, slope * xx + intercept, color='#1f77b4', lw=0.9,
                    dashes=(4, 3), zorder=1)
            # Badge top-left so it doesn't sit on the regression line for
            # negatively-correlated panels (which slope down to the bottom-right).
            ax.text(0.04, 0.96, f"{corr_label} = {corr:+.2f}\n{p_text}",
                    transform=ax.transAxes, ha='left', va='top', fontsize=5.5)
        else:
            ax.text(0.96, 0.04, f"{corr_label} = {corr:+.2f}\n{p_text}",
                    transform=ax.transAxes, ha='right', va='bottom', fontsize=5.5)
        # Panel subtitles at label size, matching the other multi-panel figures.
        ax.set_title(f"{DATASET_DISPLAY.get(name, name)} (n = {len(f)})", pad=2,
                     fontsize=plt.rcParams["axes.labelsize"])
        if use_ranks:
            n = len(f)
            ax.set_xlim(0, n + 1)
            ax.set_ylim(0, n + 1)
            ax.set_xticks([1, n // 2, n])
            ax.set_yticks([1, n // 2, n])
        else:
            ax.set_ylim(0, 1)
        # Force square panels on both variants so the two figures have
        # identical panel geometry; set_box_aspect (vs set_aspect='equal')
        # lets constrained_layout size the box to fill available space.
        ax.set_box_aspect(1)
        ax.tick_params(axis="both", length=3, pad=2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.spines["left"].set_linewidth(0.8)

    # Bottom-row x-label, left-column y-label only
    for i, ax in enumerate(axes_flat):
        ax.set_xlabel(x_label if i >= 2 else "", labelpad=3)
        ax.set_ylabel(y_label if i % 2 == 0 else "", labelpad=3)

    suptitle = f"Correlation between {metric_label} and DELTA"
    if use_ranks:
        suptitle += " (ranks)"
    fig.suptitle(suptitle)
    # Tighten constrained-layout padding so the suptitle hugs the top row and
    # the two rows sit closer together (default h_pad is generous).
    fig.get_layout_engine().set(h_pad=0.02, w_pad=0.04)
    fig.savefig(out_path, pad_inches=0.02, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote panel plot -> {out_path}")


def sweep_plot(datasets, thresholds, out_path, metric='bridging', min_n=20,
               corr_method='spearman'):
    """Sweep the min-fractional-reach threshold and plot Corr(metric, delta)
    per dataset, with Fisher-z 95% CI bands.

    `corr_method` — 'spearman' (rank) or 'pearson' (raw scores).
    Each dataset's line truncates where the kept-n drops below `min_n`.
    """
    corr_symbol = 'r' if corr_method == 'pearson' else 'ρ'
    ylabel = f"Pearson {corr_symbol}" if corr_method == 'pearson' else f"Spearman {corr_symbol}"
    set_style()
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 2.4))

    palette = sns.color_palette("tab10", n_colors=len(datasets))
    for color, name in zip(palette, datasets):
        merged = load_merged(name)
        xs, ys, lo, hi = [], [], [], []
        for t in thresholds:
            f = _filter_by_frac(merged, t)
            if len(f) < min_n:
                break
            r, l, h = _corr_with_ci(f, metric=metric, method=corr_method)
            xs.append(t); ys.append(r); lo.append(l); hi.append(h)
        ax.fill_between(xs, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.plot(xs, ys, color=color, lw=1.2,
                label=DATASET_DISPLAY.get(name, name))

    metric_label = METRIC_DISPLAY.get(metric, metric)

    ax.axhline(0, color='lightgray', lw=0.7, dashes=(4, 3), zorder=0)
    # The 0.2 cutoff used by the main-text scatter panels.
    ax.axvline(0.2, color='lightgray', lw=0.7, dashes=(4, 3), zorder=0)
    ax.set_xlabel("min fraction of group voting on comment")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Correlation between {metric_label} and DELTA")
    ax.legend(frameon=False, loc='upper center',
              bbox_to_anchor=(0.5, -0.32), ncol=2)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, pad_inches=0.02, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote sweep plot -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default=None,
                        choices=[*DATASETS, 'all'],
                        help="Dataset name, or 'all' for all four. "
                             "Required unless --sweep or --panels is set.")
    parser.add_argument('--min-votes', type=int, default=0,
                        help="Drop policies whose smaller-group vote count is below this. "
                             "Default 0 (no filter).")
    parser.add_argument('--sweep', action='store_true',
                        help="Sweep the fractional-reach threshold and plot "
                             "Corr(metric, delta) per dataset.")
    parser.add_argument('--sweep-out', default=None,
                        help="Output PDF for the sweep plot (default: {metric}_sweep[_pearson].pdf).")
    parser.add_argument('--sweep-corr', choices=['spearman', 'pearson'],
                        default='spearman',
                        help="Correlation type for the sweep plot (default spearman).")
    parser.add_argument('--panels', action='store_true',
                        help="Produce 2x2 scatter panels (per-dataset) of "
                             "DELTA-score vs metric — both score and rank versions.")
    parser.add_argument('--panels-frac', type=float, default=0.2,
                        help="min-frac threshold for the panel scatter (default 0.2).")
    parser.add_argument('--panels-metric', default='bridging_min',
                        help="Metric column to put on the y-axis of the panels.")
    parser.add_argument('--metric', default='bridging',
                        choices=['bridging', 'bridging_min', 'divergence',
                                 'in_group_support', 'in_group_support_gac',
                                 'polarization_gac', 'gac_agree',
                                 'gac_disagree', 'p_a_agree', 'p_b_agree',
                                 'upvote_rate'],
                        help="Which group-vote metric to correlate against delta.")
    args = parser.parse_args()

    if args.panels:
        ds = list(DATASETS)
        panel_plot(ds, args.panels_metric, args.panels_frac,
                   HERE / f"{args.panels_metric}_scatter_panel.pdf",
                   use_ranks=False)
        panel_plot(ds, args.panels_metric, args.panels_frac,
                   HERE / f"{args.panels_metric}_scatter_panel_ranks.pdf",
                   use_ranks=True)
        return

    if args.sweep:
        # Default sweep: fractional reach from 0 to 90% in steps of 2.5%. Lines
        # truncate where any dataset's kept-n drops below `min_n` in sweep_plot.
        thresholds = [round(0.025 * k, 3) for k in range(37)]
        suffix = '_pearson' if args.sweep_corr == 'pearson' else ''
        out = args.sweep_out or f"{args.metric}_sweep{suffix}.pdf"
        sweep_plot(list(DATASETS), thresholds, HERE / out, metric=args.metric,
                   corr_method=args.sweep_corr)
        return

    if args.dir is None:
        parser.error("--dir is required unless --sweep is set.")
    targets = list(DATASETS) if args.dir == 'all' else [args.dir]
    for d in targets:
        analyse(d, min_votes=args.min_votes)


if __name__ == '__main__':
    main()
