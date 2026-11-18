"""Panels A and B of the rest-vs-movement figure, stacked in one figure.

Panel A (top)    — bilateral power spectra per session, plus the difference
                   spectrum, from make_rest_mvt_psd_figure.py
Panel B (bottom) — the movement effect feature by feature, from
                   make_rest_mvt_forest_figure.py

THE ONLY ENTRY POINT. Both panel modules are libraries: each exposes
draw_into(fig, subspec, ...) and neither writes a file or has a CLI of its own.
This script supplies the two cells and the shared caption, and is the single
place a figure is produced — the standalone fig_rest_mvt_psd_* and
fig_rest_mvt_forest_* outputs were removed, since the panels are only ever
published together and two routes to the same graphic meant two to keep in step.

Every option that reaches a panel therefore has to be exposed here; --n-boot and
--refresh-boot are panel A's, and are the ones easiest to forget.

Why stacked
-----------
Both panels are now wide and short — A is a 2 x 4 grid (stimulation state by
session, plus the difference column), B runs its features along the x axis — so
each wants the full page width and neither is tall. Stacked they are ~9.4 x 8.4;
side by side they would be 19 inches wide. Reading order is top to bottom: A
shows where in frequency the effect sits, B summarises it band by band.

Both panels are given the same left margin, so panel A's first spectra column
and panel B's effect axis start on one vertical line.

Output: Results/Figures/fig_rest_mvt_panels_{side}.png

Usage
-----
    uv run python make_rest_mvt_combined_figure.py
    uv run python make_rest_mvt_combined_figure.py --side bilateral
    uv run python make_rest_mvt_combined_figure.py --fmax 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import make_rest_mvt_forest_figure as B
import make_rest_mvt_psd_figure as A
import table_figure_style as S
from stats_analysis.data_utils import build_psd_array, load_all_sessions

OUT_DIR = Path('Results') / 'Figures'

FIG_W     = 7.30
FIG_H     = 6.0
L_MARGIN  = 0.088   # room for panel A's tick labels, row names and axis label
R_MARGIN  = 0.955   # room for the difference column's right-hand ticks
T_MARGIN  = 0.905   # panel A's title and its first row subtitle sit above this
B_MARGIN  = 0.070
GAP_IN    = 1.15    # between the panels: panel B's title sits above its axes and
                    # panel A's frequency labels below theirs
# Panel A : panel B, as a share of the axes height left after the margins and
# GAP_IN. Panel B was shortened from 1.60 and FIG_H trimmed to match, so the
# height came off B rather than being handed to A — the ratios are relative, so
# lowering one alone would simply have made the spectra taller.
H_RATIOS  = (2.95, 1.40)


LEGEND_GAP_IN = 0.02   # between panel B's lowest tick label and its legend


def _legend_to_bottom(fig, axes_b):
    """Move panel B's Stim OFF / ON legend to the bottom centre of the figure.

    The standalone forest figure keeps it inside its first panel; here it goes
    under everything, centred on panel B's span. Placed below the lowest drawn
    tick label rather than the axes, because the 45° feature names (one of them
    wrapped onto two lines) hang well below the axes and vary in depth.
    """
    leg = axes_b[0].get_legend()
    if leg is None:
        return
    handles = list(leg.legend_handles)
    labels = [t.get_text() for t in leg.get_texts()]
    leg.remove()

    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    y_low = min(inv.transform(ax.get_tightbbox(renderer))[0, 1] for ax in axes_b)
    x0 = min(ax.get_position().x0 for ax in axes_b)
    x1 = max(ax.get_position().x1 for ax in axes_b)
    fig.legend(handles=handles, labels=labels, loc='upper center',
               bbox_to_anchor=((x0 + x1) / 2,
                               y_low - LEGEND_GAP_IN / fig.get_size_inches()[1]),
               ncol=len(handles), frameon=False, fontsize=7.0,
               handletextpad=0.5, columnspacing=2.0)


def draw(freqs, spectra, patients, df, coef, side, timepoints, fmin, fmax,
         center='geomean', show_diff=True,
         n_boot=A.N_BOOT_PSD, refresh_boot=False):
    S.apply_rcparams()

    fig = plt.figure(figsize=(FIG_W, FIG_H))

    # hspace is a fraction of the MEAN cell height, so the gap is solved for
    # rather than guessed — the two cells are deliberately unequal.
    mean_cell = sum(H_RATIOS) / len(H_RATIOS)
    gs = fig.add_gridspec(2, 1, height_ratios=list(H_RATIOS),
                          hspace=GAP_IN / mean_cell,
                          left=L_MARGIN, right=R_MARGIN,
                          top=T_MARGIN, bottom=B_MARGIN)

    # The panel letter leads its own title rather than floating in the margin:
    # as a separate mark it had to be positioned against a cell edge while the
    # title was positioned against an axis, and the two drifted apart whenever
    # either panel was resized.
    axes_a, _axes_diff, n_used, diff_note = A.draw_into(
        fig, gs[0, 0], freqs, spectra, patients, timepoints, fmin, fmax,
        center=center, show_diff=show_diff, title=f'A    {A.PANEL_TITLE}',
        n_boot=n_boot, refresh_boot=refresh_boot)

    res_b = B.draw_into(fig, gs[1, 0], df, coef, title=f'B    {B.PANEL_TITLE}')
    if res_b is None:
        plt.close(fig)
        return None
    axes_b, _panels_b, _any_boot = res_b
    _legend_to_bottom(fig, axes_b)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f'fig_rest_mvt_panels_{side}.png'
    fig.savefig(out, dpi=S.PNG_DPI, facecolor=S.SURFACE, bbox_inches='tight')
    if side in ('L', 'bilateral'):   # the sides with a TIFF requested
        fig.savefig(out.with_suffix('.tiff'), dpi=S.PNG_DPI, facecolor=S.SURFACE,
                    bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
    return out, n_used, diff_note


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--side', default='bilateral')
    ap.add_argument('--fmin', type=float, default=1.0)
    ap.add_argument('--fmax', type=float, default=45.0)
    ap.add_argument('--timepoints', nargs='*', default=A.DEFAULT_TPS)
    ap.add_argument('--center', choices=['geomean', 'median'], default='geomean')
    ap.add_argument('--no-diff', dest='diff', action='store_false',
                    help='drop panel A\'s difference-spectrum row')
    ap.add_argument('--n-boot', type=int, default=A.N_BOOT_PSD,
                    help=f'cluster-bootstrap resamples for panel A\'s difference '
                         f'ribbon (default {A.N_BOOT_PSD}); 0 draws it with no ribbon')
    ap.add_argument('--refresh-boot', action='store_true',
                    help='recompute that bootstrap instead of reusing the cache')
    args = ap.parse_args(argv)

    df, coef, path = B.load_side(args.side)
    if df is None:
        print(f'  [missing] {path}')
        print('\nRun the analysis first:')
        print(f'  uv run python stats_analysis/lme_rest_mvt_diff.py --sides {args.side}')
        return 1

    all_sessions = load_all_sessions()
    freqs, spectra, patients = build_psd_array(all_sessions, side=args.side)

    res = draw(freqs, spectra, patients, df, coef, args.side, args.timepoints,
               args.fmin, args.fmax, center=args.center, show_diff=args.diff,
               n_boot=args.n_boot, refresh_boot=args.refresh_boot)
    if res is None:
        print('  [failed] no forest panel could be laid out')
        return 1

    png, n_used, diff_note = res
    print(f'  panel A rows: {", ".join(args.timepoints)}')
    for (tp, cond), n in sorted(n_used.items()):
        print(f'    {tp:<5s} {cond:<16s} n = {n} patients')
    if diff_note:
        print(f'  {diff_note}')
    print(f'  Saved -> {png}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
