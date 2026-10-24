"""Convergence heatmap: for each (n_cases, n_raters) cell, draw N_REPS fresh
subsamples (independent of other cells) and compute Spearman ρ of the
subsample-DELTA against the full-data DELTA. Each cell shows the mean ρ over
those reps.

Expect: low correlation in the bottom-left (few cases, few raters), saturating
toward 1.0 in the top-right.

Usage:
    python analysis/convergence/case_rater_heatmap.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from delta.scoring import disagreement, revealed_probability
from delta import load_probs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_style import FIG_WIDTH, set_style

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

# name -> (results_dir, max_cases, title_suffix). max_cases=None means use all.
DATASETS = {
    'human_experiment': ('results/human_experiment',       None, 'human experiment'),
    'vtaiwan':          ('results/vtaiwan',                 None, 'Polis - Uber'),
}

N_REPS = 100
SEED = 42


def build_revealed_array(probs):
    """Pivot the long probs frame into a (policy, case, rater) array of the
    revealed-yes probability  p_revealed = (p_comply + (1 - p_violate)) / 2.
    """
    comply = (probs[probs['prompt'] == 0]
              .set_index(['policy', 'case', 'rater'])['prob_yes']
              .rename('p_comply'))
    violate = (probs[probs['prompt'] == 1]
               .set_index(['policy', 'case', 'rater'])['prob_yes']
               .rename('p_violate'))
    revealed = revealed_probability(comply, violate).rename('p').reset_index()

    policies = sorted(revealed['policy'].unique())
    cases    = sorted(revealed['case'].unique())
    raters   = sorted(revealed['rater'].unique())
    p_idx = {p: i for i, p in enumerate(policies)}
    c_idx = {c: i for i, c in enumerate(cases)}
    r_idx = {r: i for i, r in enumerate(raters)}

    arr = np.full((len(policies), len(cases), len(raters)), np.nan)
    arr[
        revealed['policy'].map(p_idx).to_numpy(),
        revealed['case'].map(c_idx).to_numpy(),
        revealed['rater'].map(r_idx).to_numpy(),
    ] = revealed['p'].to_numpy()
    return arr, policies, cases, raters


def delta_from_subarr(sub):
    """Per-policy DELTA score from a (policy, case, rater) subarray.

    Cell score = 4 * mean_p * (1 - mean_p) over raters.
    Policy DELTA = mean of cell scores across cases.
    """
    with np.errstate(invalid='ignore'):
        mean = np.nanmean(sub, axis=2)
        cell = disagreement(mean)
        return np.nanmean(cell, axis=1)


def spearman(a, b):
    # Tiny subsamples can produce constant rank vectors (all policies tied), in
    # which case corrcoef divides by zero and emits NaN — return NaN explicitly
    # and let nanmean handle it at the cell-aggregation step.
    with np.errstate(invalid='ignore', divide='ignore'):
        return float(np.corrcoef(pd.Series(a).rank(), pd.Series(b).rank())[0, 1])


def run_dataset(name, results_dir, max_cases, title_suffix):
    probs = load_probs(ROOT / results_dir, 'probs.csv')
    if max_cases is not None:
        keep = sorted(probs['case'].unique())[:max_cases]
        probs = probs[probs['case'].isin(keep)]
        print(f"{name}: truncated to first {len(keep)} cases (of original)")
    arr, policies, cases, raters = build_revealed_array(probs)
    n_pol, n_case_max, n_rater_max = arr.shape
    print(f"loaded {name}: {n_pol} policies, {n_case_max} cases, {n_rater_max} raters")

    ref_delta = delta_from_subarr(arr)
    ref_ranks = pd.Series(ref_delta).rank().to_numpy()

    case_grid  = list(range(2, n_case_max + 1))
    rater_grid = list(range(2, n_rater_max + 1))

    heatmap = np.zeros((len(rater_grid), len(case_grid)))
    rng = np.random.default_rng(SEED)
    case_idx_all  = np.arange(n_case_max)
    rater_idx_all = np.arange(n_rater_max)
    for ri, n_r in enumerate(rater_grid):
        for ci, n_c in enumerate(case_grid):
            rhos = np.empty(N_REPS)
            for k in range(N_REPS):
                # Fresh subsample for every rep — no reuse across cells or reps.
                cc = rng.choice(case_idx_all,  size=n_c, replace=False)
                cr = rng.choice(rater_idx_all, size=n_r, replace=False)
                sub = arr[:, cc, :][:, :, cr]
                sub_delta = delta_from_subarr(sub)
                rhos[k] = spearman(ref_ranks, sub_delta)
            heatmap[ri, ci] = float(np.nanmean(rhos))
        print(f"  n_raters={n_r:>2}  row mean ρ = {heatmap[ri].mean():+.3f}")

    # Numeric output: rows = n_raters, cols = n_cases
    df_out = pd.DataFrame(heatmap, index=rater_grid, columns=case_grid)
    df_out.index.name  = 'n_raters'
    df_out.columns.name = 'n_cases'
    csv_out = HERE / f'case_rater_heatmap_{name}.csv'
    df_out.to_csv(csv_out)
    print(f"Wrote {csv_out}")

    # The canvas stays FIG_WIDTH wide so fonts print at the shared size, but
    # the grid itself is deliberately smaller: axes are placed by hand at
    # about half the canvas width, and the figure is saved WITHOUT a tight
    # bounding box, since tight-cropping would trim the margins and shrink
    # the final PDF (magnifying the fonts at \columnwidth).
    fig = plt.figure(figsize=(FIG_WIDTH, 1.9))
    ax = fig.add_axes([0.21, 0.22, 0.50, 0.58])
    cax = fig.add_axes([0.74, 0.22, 0.025, 0.58])
    extent = [case_grid[0]  - 0.5, case_grid[-1]  + 0.5,
              rater_grid[0] - 0.5, rater_grid[-1] + 0.5]
    vmin = float(heatmap.min())
    im = ax.imshow(heatmap, origin='lower', aspect='auto',
                   extent=extent, cmap='RdYlGn', vmin=vmin, vmax=1.0)

    ax.set_xlim(case_grid[0]  - 0.5, case_grid[-1]  + 0.5)
    ax.set_ylim(rater_grid[0] - 0.5, rater_grid[-1] + 0.5)
    ax.set_xlabel("number of cases")
    ax.set_ylabel("number of raters")
    ax.set_title(f"LLM-DELTA score convergence — {title_suffix}")

    cbar = fig.colorbar(im, cax=cax)
    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(width=0.5, length=2,
                        labelsize=plt.rcParams["xtick.labelsize"])
    # Heatmap: keep all four spines (full border), don't despine.
    out = HERE / f'case_rater_heatmap_{name}.pdf'
    # CreationDate=None keeps the PDF byte-identical across runs, so a
    # re-render does not show up as a spurious diff. No bbox_inches="tight"
    # here -- see the canvas comment above.
    fig.savefig(out, metadata={"CreationDate": None})
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    set_style()
    for name, (results_dir, max_cases, title_suffix) in DATASETS.items():
        run_dataset(name, results_dir, max_cases, title_suffix)


if __name__ == "__main__":
    main()
