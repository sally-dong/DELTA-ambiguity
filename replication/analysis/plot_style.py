"""One styling contract for every paper figure.

All figures share seaborn's standard sans-serif theme and are rendered
FIG_WIDTH inches wide, so that when the paper includes each one at
\\columnwidth every font lands at the same printed size. Figure-level
titles are 8pt everywhere -- both single-axes titles and multi-panel
suptitles -- and per-panel subtitles drop to the axis-label size.

Usage, from any script in this repository:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from plot_style import FIG_WIDTH, set_style
"""

import seaborn as sns

# Width in inches of every figure; include at \columnwidth in the paper.
FIG_WIDTH = 3.3


def set_style():
    """Apply the shared seaborn theme. Call before creating any figure."""
    sns.set_theme(
        style="ticks",
        context="paper",
        font_scale=0.75,
        rc={
            "axes.titlesize": 8,
            "axes.titleweight": "regular",
            "figure.titlesize": 8,
        },
    )
