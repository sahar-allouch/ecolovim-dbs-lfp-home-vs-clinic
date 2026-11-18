"""Monopolar impedance per contact, M3 → M6.

A 2 x 9 grid: one row per hemisphere, one column per contact, with the mean
across contacts carried in the leftmost column so the summary sits beside the
detail it summarises rather than in a separate plate. Every panel shares one
impedance axis, so a contact that sits high does so visibly.

Each panel draws one faint line per patient (M3 to M6) with the cohort median
solid and the mean dashed over them — the median is what the Wilcoxon tests
report, the mean is shown because impedance is quoted as a mean in the device
literature and the two can disagree at n = 9.

The Wilcoxon results are read from the existing analysis, not recomputed: the
per-contact panels are annotated with the FDR-corrected P value from
wilcoxon_impedance.py and the mean-across-contacts panels with its
hemisphere-level P.

Output: Results/Figures/impedance_per_contact_grid.png
Run:    uv run python make_impedance_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

matplotlib.use('Agg')

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import table_figure_style as S  # noqa: E402

CSV = ROOT / 'Results/Tables/impedance_all_patients.csv'
WILCOXON = ROOT / 'Results/stats/wilcoxon_impedance/wilcoxon_impedance_M3_M6.xlsx'
OUT = ROOT / 'Results/Figures'

# ── Content ──────────────────────────────────────────────────────────────────
CONTACT_ORDER = ['0', '1a', '1b', '1c', '2a', '2b', '2c', '3']
HEMIS = ['Left', 'Right']
TIMEPOINTS = ['M3', 'M6']
MODE = 'Monopolar'

# ── Palette ──────────────────────────────────────────────────────────────────
# Hemisphere is the only categorical variable inside this figure — the two
# sessions are positions on the x axis, not colours — so the study hues are free
# to carry it here without clashing with their session meaning elsewhere.
C_HEMI = {'Left': '#355C9A', 'Right': '#C95C54'}

INK = '#1F2328'
MUTED = '#6B7280'
FAINT = '#E5E7EB'
SURFACE = '#ffffff'

SZ_HEAD = 7.5       # subplot titles (patient header, per-contact/mean), bold
SZ_LABEL = 7.5      # axis labels
SZ_SMALL = 7.0      # legend
SZ_TICK = 7.0        # tick labels
SZ_TINY = 6.5       # small annotations (significance markers)

LW_PATIENT = 0.6
LW_MEDIAN = 1.7
LW_MEAN = 1.1
ALPHA_PATIENT = 0.22
DASH_MEAN = (0, (3.0, 1.3))

ALPHA_SIG = 0.05


def _lighten(hex_color: str, amount: float) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    r, g, b = (int(round(c + (255 - c) * amount)) for c in (r, g, b))
    return f'#{r:02X}{g:02X}{b:02X}'


# ── Data ─────────────────────────────────────────────────────────────────────

def load_impedance() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per contact, mean across contacts) — both wide on timepoint.

    Repeat sessions at one timepoint are averaged first, so a patient recorded
    twice at M3 still contributes one M3 value and cannot outweigh the others.
    """
    df = pd.read_csv(CSV)
    mono = df[df['Mode'] == MODE].copy()
    mono['Contact'] = mono['Electrode2'].str.replace(r'(?i)sensight_', '',
                                                     regex=True)
    mono = mono[mono['Timepoint'].isin(TIMEPOINTS)]

    per_contact = (mono
                   .groupby(['Patient', 'Timepoint', 'Hemisphere', 'Contact'])
                   ['Impedance_Ohm'].mean().reset_index())
    wide = (per_contact
            .pivot_table(index=['Patient', 'Hemisphere', 'Contact'],
                         columns='Timepoint', values='Impedance_Ohm')
            .reset_index().dropna(subset=TIMEPOINTS))

    per_hemi = (mono.groupby(['Patient', 'Timepoint', 'Hemisphere'])
                ['Impedance_Ohm'].mean().reset_index())
    wide_hemi = (per_hemi
                 .pivot_table(index=['Patient', 'Hemisphere'],
                              columns='Timepoint', values='Impedance_Ohm')
                 .reset_index().dropna(subset=TIMEPOINTS))
    return wide, wide_hemi


def load_wilcoxon() -> tuple[dict, dict]:
    """{(hemisphere, contact): p_fdr} and {hemisphere: p_fdr}.

    Both levels are BH-corrected within their own family (16 contact tests,
    2 hemisphere tests), so both annotations report the corrected p-value.

    Empty when the analysis has not been run — the panels then carry no P
    annotation rather than a recomputed one.
    """
    if not WILCOXON.exists():
        return {}, {}
    per_contact = pd.read_excel(WILCOXON, sheet_name='Per_Contact_FDR')
    per_hemi = pd.read_excel(WILCOXON, sheet_name='Mean_per_Hemisphere')
    contact_p = {(str(r.Hemisphere), str(r.Contact)): float(r.p_fdr)
                 for r in per_contact.itertuples()}
    hemi_p = {}
    for r in per_hemi.itertuples():
        for hemi in HEMIS:
            if hemi.lower() in str(r.label).lower():
                hemi_p[hemi] = float(r.p_fdr)
    return contact_p, hemi_p


# ── Drawing ──────────────────────────────────────────────────────────────────

def fmt_p(p: float | None) -> str:
    """Italic lower-case p, upright value — the usual statistical convention.

    Only the symbol is set in maths italic; the number, operator and asterisk
    stay upright.
    """
    if p is None or not np.isfinite(p):
        return ''
    star = '*' if p < ALPHA_SIG else ''
    value = '< 0.001' if p < 0.001 else f'= {p:.3f}'
    return f'$\\it{{p}}$ {value}{star}'


def draw_panel(ax, sub: pd.DataFrame, hemi: str, p_value, ylim,
               first_col: bool, last_row: bool):
    """One slope panel: patients behind, IQR band, then median and mean.

    Only the leftmost panel of a row carries a y axis and only the bottom row
    carries session labels — the grid shares both, and repeating them eighteen
    times is ink that adds nothing.
    """
    color = C_HEMI[hemi]
    faint = _lighten(color, 0.50)
    x = [0, 1]

    if len(sub):
        # IQR band: the spread the median is drawn from, shown rather than
        # implied. Two timepoints make it a quadrilateral, not a ribbon.
        q1 = [sub['M3'].quantile(0.25) / 1000, sub['M6'].quantile(0.25) / 1000]
        q3 = [sub['M3'].quantile(0.75) / 1000, sub['M6'].quantile(0.75) / 1000]
        ax.fill_between(x, q1, q3, color=color, alpha=0.10, lw=0, zorder=1)

    for _, r in sub.iterrows():
        ax.plot(x, [r['M3'] / 1000, r['M6'] / 1000], color=faint,
                lw=LW_PATIENT, alpha=ALPHA_PATIENT, solid_capstyle='round',
                zorder=2)

    if len(sub):
        ax.plot(x, [sub['M3'].mean() / 1000, sub['M6'].mean() / 1000],
                color=color, lw=LW_MEAN, ls=DASH_MEAN, zorder=3)
        ax.plot(x, [sub['M3'].median() / 1000, sub['M6'].median() / 1000],
                color=color, lw=LW_MEDIAN, marker='o', ms=3.4,
                markeredgecolor=SURFACE, markeredgewidth=0.6, zorder=4)

    ax.set_xlim(-0.30, 1.30)
    ax.set_ylim(*ylim)
    ax.set_xticks(x)
    ax.set_xticklabels(TIMEPOINTS if last_row else ['', ''], fontsize=SZ_TICK)
    ax.tick_params(axis='x', length=0, pad=3)
    ax.tick_params(axis='y', labelsize=SZ_TICK, length=2.4, width=0.8,
                   color=INK, pad=1.5)
    ax.grid(True, axis='y', color=FAINT, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    # Y axis on the first panel only; the rest are bounded by the shared grid,
    # which keeps eight repeated spines out of the plate.
    for name, sp in ax.spines.items():
        sp.set_visible(first_col and name == 'left')
        sp.set_color(INK)
        sp.set_linewidth(0.8)
    if not first_col:
        ax.tick_params(axis='y', length=0, labelleft=False)

    # P sits inside the panel, under the column header: below the axis it would
    # collide with the session labels on the bottom row.
    txt = fmt_p(p_value)
    if txt:
        sig = txt.endswith('*')
        ax.text(0.5, 0.99, txt, transform=ax.transAxes, ha='center', va='top',
                fontsize=SZ_TINY, color=INK if sig else MUTED,
                fontweight='bold' if sig else 'normal', zorder=5)


def build():
    S.apply_rcparams()
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE, 'text.color': INK,
        'axes.labelcolor': INK, 'axes.edgecolor': INK,
        'xtick.color': INK, 'ytick.color': INK,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        # Maths text in the figure's own face, so the italic p in the P
        # annotations is Arial italic rather than matplotlib's default serif-ish
        # fallback.
        'mathtext.fontset': 'custom',
        'mathtext.rm': S.FAMILY,
        'mathtext.it': f'{S.FAMILY}:italic',
        'mathtext.default': 'regular',
    })

    wide, wide_hemi = load_impedance()
    contact_p, hemi_p = load_wilcoxon()

    # One impedance axis for the whole grid: the panels are only comparable if
    # a value at the same height means the same thing in each of them.
    vals = pd.concat([wide['M3'], wide['M6'], wide_hemi['M3'],
                      wide_hemi['M6']]) / 1000       # kΩ: shorter tick labels
    pad = 0.07 * (vals.max() - vals.min())
    # Extra headroom at the top: the P annotation lives inside the panel.
    ylim = (max(0.0, vals.min() - pad), vals.max() + 3.2 * pad)

    n_col = len(CONTACT_ORDER) + 1
    # Column 0 is a different kind of panel, so it sits in its own width-ratio
    # slot with a wider gap after it; the gap does the separating, not a rule.
    widths = [1.15] + [1.0] * len(CONTACT_ORDER)
    left, right = 0.095, 0.988
    fig = plt.figure(figsize=(7.28, 3.6))
    gs = fig.add_gridspec(2, n_col, width_ratios=widths, hspace=0.26,
                          wspace=0.20, left=left, right=right,
                          top=0.885, bottom=0.175)

    for row, hemi in enumerate(HEMIS):
        last = row == len(HEMIS) - 1
        # Column 0: mean across contacts, the same summary the app's grid shows.
        ax = fig.add_subplot(gs[row, 0])
        draw_panel(ax, wide_hemi[wide_hemi['Hemisphere'] == hemi], hemi,
                   hemi_p.get(hemi), ylim, first_col=True, last_row=last)
        ax.set_ylabel('Impedance (kΩ)', fontsize=SZ_LABEL, labelpad=2)
        if row == 0:
            ax.set_title('Mean of contacts', fontsize=SZ_HEAD, color=MUTED,
                         fontstyle='italic', pad=5)
        # Hemisphere label outboard of the axis label, in the row's own colour.
        ax.text(-0.72, 0.5, hemi, transform=ax.transAxes, rotation=90,
                ha='center', va='center', fontsize=SZ_HEAD, fontweight='bold',
                color=C_HEMI[hemi])

        sub_hemi = wide[wide['Hemisphere'] == hemi]
        for col, contact in enumerate(CONTACT_ORDER, start=1):
            ax = fig.add_subplot(gs[row, col])
            draw_panel(ax, sub_hemi[sub_hemi['Contact'] == contact], hemi,
                       contact_p.get((hemi, contact)), ylim, first_col=False,
                       last_row=last)
            if row == 0:
                ax.set_title(contact, fontsize=SZ_HEAD, color=INK,
                             fontweight='bold', pad=5)

    n_pt = wide_hemi['Patient'].nunique()
    handles = [
        Line2D([], [], color=MUTED, lw=LW_MEDIAN, marker='o', ms=3.4,
               markeredgecolor=SURFACE, markeredgewidth=0.6, label='Median'),
        Line2D([], [], color=MUTED, lw=LW_MEAN, ls=DASH_MEAN, label='Mean'),
        Patch(facecolor=MUTED, alpha=0.18, label='Interquartile range'),
        Line2D([], [], color=MUTED, lw=LW_PATIENT, alpha=0.55,
               label=f'Individual patients (n = {n_pt})'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False,
               fontsize=SZ_SMALL, handlelength=1.9, columnspacing=1.8,
               handletextpad=0.6, bbox_to_anchor=(0.5, -0.012))

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / 'impedance_per_contact_grid.png', dpi=S.PNG_DPI,
                facecolor=SURFACE)
    plt.close(fig)

    print(f'saved {OUT / "impedance_per_contact_grid.png"}')
    print(f'  patients      : {n_pt}')
    print(f'  contacts      : {", ".join(CONTACT_ORDER)}')
    print(f'  impedance     : {vals.min() * 1000:.0f}{S_ENDASH}'
          f'{vals.max() * 1000:.0f} Ω')
    print('  Wilcoxon      : '
          + ('read from ' + WILCOXON.name if contact_p else 'not available'))
    if contact_p:
        sig = sorted(k for k, p in contact_p.items() if p < ALPHA_SIG)
        print('  FDR-significant contacts: '
              + (', '.join(f'{h} {c}' for h, c in sig) if sig else 'none'))


S_ENDASH = '–'


if __name__ == '__main__':
    build()
