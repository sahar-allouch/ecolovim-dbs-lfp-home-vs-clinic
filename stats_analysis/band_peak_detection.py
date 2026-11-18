"""
band_peak_detection.py
======================
Count FOOOF band-peak detections per band x condition x timepoint x side.

A peak counts as "detected" when `get_band_peak_fm` returned a non-NaN center
frequency for that band, i.e. `band_peaks_allch[band][0, ch_idx, 0]` is finite.
The channel is the patient's stimulation contact on the
requested side (STIM_CHANNELS), so one value per patient/side/band/condition.

The PSD is mean-reduced across epochs before the FOOOF fit, so
`band_peaks_allch[band]` has a single epoch (axis 0 of length 1); a longer axis
means the upstream reduction changed and raises.

Every patient with a usable stimulation contact is counted, so the denominator
is per cell and varies: 10 at M3, 8 (stim-OFF) or 9 (stim-ON) at home, 9 at M6.
P03 has no data after M3; P04 has no at-home stim-OFF recordings. The
denominator is printed in every cell (k/n) and reported on its own in
`availability.csv`, so no patient is dropped to equalise n.

Column layout: L timepoints (M3, home, M6) then R timepoints (M3, home, M6).

Run from the repo root:
    uv run python stats_analysis/band_peak_detection.py
    uv run python stats_analysis/band_peak_detection.py --bands alpha beta theta
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from config import BAND_LABELS  # noqa: E402
from stats_analysis.data_utils import (  # noqa: E402
    CONDITIONS,
    NB_PATIENTS,
    STIM_CHANNELS,
    TIMEPOINTS,
    load_all_sessions,
)

SIDES       = ['L', 'R']
DEFAULT_BANDS = ['alpha', 'beta']
OUT_DIR     = Path(__file__).parent.parent / 'Results' / 'stats' / 'band_peaks'

# L: M3, home, M6  |  R: M3, home, M6
COL_ORDER = pd.MultiIndex.from_tuples(
    [(side, tp) for side in SIDES for tp in TIMEPOINTS], names=['side', 'timepoint']
)


def peak_detected(feat, patient_idx, side, band):
    """True/False if a peak was/wasn't detected, None if the patient has no data."""
    if feat is None:
        return None
    ch_name = STIM_CHANNELS.get(patient_idx, {}).get(side)
    ch_names = list(feat.get('ch_names', []))
    if ch_name is None or ch_name not in ch_names:
        return None
    band_peaks = feat.get('band_peaks_allch')
    if band_peaks is None or band_peaks.get(band) is None:
        return None

    arr = np.asarray(band_peaks[band])           # (n_epochs, n_ch, 3) = [cf, pw, bw]
    if arr.shape[0] != 1:
        raise ValueError(
            f"band_peaks_allch['{band}'] has {arr.shape[0]} epochs for "
            f"P{patient_idx + 1}/{side} (expected 1 — the PSD is mean-reduced "
            "across epochs before the FOOOF fit)."
        )
    ch_idx = ch_names.index(ch_name)
    if ch_idx >= arr.shape[1]:
        return None
    return bool(np.isfinite(arr[0, ch_idx, 0]))   # cf


def build_long_df(all_sessions, bands):
    """One row per band x condition x timepoint x side x patient."""
    rows = []
    for band in bands:
        for condition in CONDITIONS:
            for timepoint in TIMEPOINTS:
                patients = all_sessions[timepoint].get(condition, [])
                for p in range(NB_PATIENTS):
                    feat = patients[p] if p < len(patients) else None
                    for side in SIDES:
                        det = peak_detected(feat, p, side, band)
                        rows.append({
                            'band':      band,
                            'condition': condition,
                            'timepoint': timepoint,
                            'side':      side,
                            'patient':   p + 1,
                            'available': det is not None,
                            'detected':  det,
                        })
    return pd.DataFrame(rows)


def _cells_with_pct(grp):
    """'k/n (p%)' strings from a frame holding `detected` and `available`."""
    pct = (100 * grp.detected / grp.available.replace(0, np.nan)).round(0)
    pct = pct.fillna(0).astype(int)
    return (grp.detected.astype(str) + '/' + grp.available.astype(str)
            + ' (' + pct.astype(str) + '%)')


def counts_table(df):
    """Wide 'k/n (p%)' table — rows = condition, cols = (side, timepoint).

    n is the number of patients with a usable channel in that cell; nobody is
    dropped to make n equal across timepoints, so the percentage is what makes
    cells comparable across timepoints.
    """
    grp = (df.groupby(['condition', 'side', 'timepoint'])
             .agg(detected=('detected', lambda s: int(s.fillna(False).sum())),
                  available=('available', 'sum'))
             .reset_index())
    grp['cell'] = _cells_with_pct(grp)
    wide = grp.pivot(index='condition', columns=['side', 'timepoint'], values='cell')
    return wide.reindex(index=CONDITIONS, columns=COL_ORDER)


def pooled_table(df):
    """Detections pooled over the four conditions — rows = band, cols = (side, timepoint)."""
    grp = (df.groupby(['band', 'side', 'timepoint'])
             .agg(detected=('detected', lambda s: int(s.fillna(False).sum())),
                  available=('available', 'sum'))
             .reset_index())
    grp['cell'] = _cells_with_pct(grp)
    wide = grp.pivot(index='band', columns=['side', 'timepoint'], values='cell')
    return wide.reindex(index=list(df.band.unique()), columns=COL_ORDER)


def availability_table(df):
    """Patients with a usable channel — rows = condition, cols = (side, timepoint)."""
    grp = (df.groupby(['condition', 'side', 'timepoint'])['available']
             .sum().reset_index())
    # same across bands, so divide by the number of bands
    grp['available'] = grp['available'] // df.band.nunique()
    wide = grp.pivot(index='condition', columns=['side', 'timepoint'], values='available')
    return wide.reindex(index=CONDITIONS, columns=COL_ORDER)


def missing_report(df):
    """Which patients are missing where — rows = condition x timepoint x side."""
    miss = df[~df.available].drop_duplicates(['condition', 'timepoint', 'side', 'patient'])
    if miss.empty:
        return pd.DataFrame(columns=['condition', 'timepoint', 'side', 'missing_patients'])
    out = (miss.groupby(['condition', 'timepoint', 'side'])['patient']
               .apply(lambda s: ', '.join(f'P{p:02d}' for p in sorted(s)))
               .reset_index(name='missing_patients'))
    return out


def _fmt(wide):
    """Flatten the (side, timepoint) columns to 'L M3' style for printing/export."""
    out = wide.copy()
    out.columns = [f'{side} {tp}' for side, tp in out.columns]
    return out.reset_index()


# ── Article table figures ─────────────────────────────────────────────────────
# Booktabs-style: no vertical rules, three horizontal rules, spanning side
# headers. Drawn manually rather than with plt.table so the L/R group headers
# can span their three timepoint columns.
FIG_FONT   = ['Arial', 'DejaVu Sans']
FIG_INK    = '#222222'
FIG_MID    = '#555555'
FIG_RULE   = '#222222'
ROW_H      = 0.30      # inches per row
FILL_H     = 0.70 * ROW_H   # height of a cell background fill (see cell_colors)
LABEL_W    = 2.6       # inches for the row-label column
COL_W      = 1.15      # inches per data column — fits 'k/n (p%)' cells


POOLED_ROW_LABEL = 'All conditions'


FIG_FORMATS = ('png',)   # one sub-folder per extension


def render_table_figure(sections, col_labels, group_labels, fig_dir, name,
                        label_header='Condition', title=None, footnote=None,
                        midrule_before=(), flat=False, col_w=COL_W,
                        label_w=LABEL_W, cell_colors=None, scale=1.0,
                        legend=None, font=None):
    """Render a plain article-ready table, one sub-folder per file format.

    sections      — [(section_title | None, [(row_label, [cell, ...]), ...]), ...]
    col_labels    — sub-header per data column, e.g. ['M3', 'home', 'M6'] * 2
    group_labels  — [(text, first_col, last_col), ...] spanning headers (L / R)
    fig_dir       — parent directory; files go to fig_dir/<ext>/<name>.<ext>
    midrule_before— row labels to precede with a thin rule (e.g. the pooled row)
    flat          — write fig_dir/<name>.<ext> instead, next to the other figures
    col_w/label_w — column widths in inches, for tables whose cells are wider
                    or narrower than the 'k/n (p%)' this was first sized for.
                    col_w is a scalar for uniform columns, or a sequence of one
                    width per column when the cells differ enough in length
                    that a single width wastes page space
    cell_colors   — {(data_row, col): facecolor} background fills; data_row
                    counts body rows across all sections, section titles excluded
    scale         — zooms the whole plate (column widths, row pitch, every font
                    size and rule width) by this factor, so a caller can land at
                    a target print width without packing the native 9 pt text
                    into narrower columns than it was sized for. 1.0 (default)
                    is the native size every other caller of this function uses.
    legend        — [(color, label), ...] swatches drawn in one row under the
                    bottom rule, for tables whose cell_colors need a key. None
                    (default) draws nothing, so existing callers are unaffected.
    font          — font family list for this call only. None (default) uses
                    the shared FIG_FONT, so a caller that wants a different
                    typeface does not change every other table that shares this
                    renderer.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties

    def _z(v):
        return v * scale

    _font_prop = FontProperties(family=font if font is not None else FIG_FONT)

    def _text_w(s, pt):
        """Rendered width of s at pt (points), in inches — actual glyph
        metrics, not a per-character guess, so a legend/footnote long enough
        to be the widest thing on the plate still gets a correctly sized
        canvas instead of running off whatever width the table happened to
        need."""
        return TextPath((0, 0), s, size=pt, prop=_font_prop).get_extents().width / 72.0

    plt.rcParams['font.family']  = font if font is not None else FIG_FONT
    plt.rcParams['pdf.fonttype'] = 42        # editable text in the PDF

    row_h  = _z(ROW_H)
    fill_h = 0.70 * row_h

    n_cols = len(col_labels)
    n_body = sum(len(rows) + (1 if sec else 0) for sec, rows in sections)
    n_head = 2                                    # group row + column row
    n_rows = n_head + n_body

    # col_w is a scalar for uniform columns, or one width per column when the
    # cells differ in length enough that a single width wastes the page.
    if isinstance(col_w, (int, float)):
        widths = [_z(float(col_w))] * n_cols
    else:
        widths = [_z(float(w)) for w in col_w]
        if len(widths) != n_cols:
            raise ValueError(f'col_w has {len(widths)} widths for '
                             f'{n_cols} columns')
    label_w = _z(label_w)

    # Left edge of every column, plus the right edge of the last.
    edges = [label_w]
    for w in widths:
        edges.append(edges[-1] + w)

    fig_w = edges[-1]

    # A legend wide enough to be the widest thing on the plate must set fig_w
    # itself — it used to just draw at whatever width the table needed and
    # ran off the right edge once a caller's table got narrower than its own
    # legend.
    _legend_sw, _legend_gap, _legend_pad = _z(0.16), _z(0.28), _z(0.10)
    if legend:
        legend_w = _legend_pad + sum(
            _legend_sw + _z(0.06) + _text_w(lab, _z(8)) + _legend_gap
            for _, lab in legend) - _legend_gap + _legend_pad
        fig_w = max(fig_w, legend_w)

    fig_h = (n_rows + 1.2) * row_h
    fig_h += _z(0.35) if title else 0
    fig_h += _z(0.35) if footnote else 0
    fig_h += _z(0.30) if legend else 0

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis('off')

    x_label = _z(0.12)
    x_cols  = [(edges[i] + edges[i + 1]) / 2 for i in range(n_cols)]
    x_left, x_right = _z(0.05), fig_w - _z(0.05)

    y = fig_h - (_z(0.45) if title else _z(0.15))
    if title:
        ax.text(x_left, y, title, ha='left', va='bottom',
                fontsize=_z(10), fontweight='bold', color=FIG_INK)
        y -= _z(0.12)

    def rule(y_pos, x0=x_left, x1=x_right, lw=1.0):
        ax.add_line(Line2D([x0, x1], [y_pos, y_pos], color=FIG_RULE,
                           lw=lw, solid_capstyle='butt'))

    rule(y, lw=_z(1.2))                            # top rule
    y -= row_h

    # Spanning group headers (L / R), each underlined over its own columns
    for text, c0, c1 in group_labels:
        xc = (x_cols[c0] + x_cols[c1]) / 2
        ax.text(xc, y + _z(0.07), text, ha='center', va='bottom',
                fontsize=_z(9), color=FIG_INK)
        rule(y + _z(0.02), edges[c0] + _z(0.05), edges[c1 + 1] - _z(0.05),
             lw=_z(0.6))
    y -= row_h

    ax.text(x_label, y + _z(0.07), label_header, ha='left', va='bottom',
            fontsize=_z(9), color=FIG_INK)
    for x, lab in zip(x_cols, col_labels):
        ax.text(x, y + _z(0.07), lab, ha='center', va='bottom',
                fontsize=_z(9), color=FIG_INK)
    y -= _z(0.06)
    rule(y, lw=_z(1.0))                            # header rule
    y -= row_h * 0.55

    data_row = 0                      # index into cell_colors, section rows skipped
    for sec_title, rows in sections:
        if sec_title:
            ax.text(x_label, y - _z(0.06), sec_title, ha='left', va='center',
                    fontsize=_z(9), fontstyle='italic', color=FIG_INK)
            y -= row_h
        for row_label, cells in rows:
            if row_label in midrule_before:
                y += row_h * 0.20
                rule(y, lw=_z(0.6))
                y -= row_h * 0.20 + row_h * 0.10
            ax.text(x_label + (_z(0.18) if sec_title else 0), y - _z(0.06),
                    row_label, ha='left', va='center', fontsize=_z(9),
                    color=FIG_INK)
            for c, (x, cell) in enumerate(zip(x_cols, cells)):
                fill = cell_colors.get((data_row, c)) if cell_colors else None
                if fill:
                    # Behind the text (zorder 0). Height is inset to fill_h: a
                    # full row_h patch on the last row would reach past the
                    # bottom rule, which sits 0.55*row_h below the last baseline.
                    ax.add_patch(Rectangle(
                        (edges[c] + _z(0.02), y - _z(0.06) - fill_h / 2),
                        widths[c] - _z(0.04), fill_h, facecolor=fill,
                        edgecolor='none', zorder=0))
                ax.text(x, y - _z(0.06), cell, ha='center', va='center',
                        fontsize=_z(9), color=FIG_INK)
            data_row += 1
            y -= row_h

    y += row_h * 0.45
    rule(y, lw=_z(1.2))                            # bottom rule

    if legend:
        lx = x_left
        ly = y - _z(0.30)
        for color, label in legend:
            ax.add_patch(Rectangle((lx, ly - _legend_sw / 2), _legend_sw, _legend_sw,
                                   facecolor=color, edgecolor='none'))
            ax.text(lx + _legend_sw + _z(0.06), ly, label, ha='left', va='center',
                    fontsize=_z(8), color=FIG_INK)
            lx += _legend_sw + _z(0.06) + _text_w(label, _z(8)) + _legend_gap
        y = ly - _z(0.20)

    if footnote:
        ax.text(x_left, y - _z(0.02), footnote, ha='left', va='top',
                fontsize=_z(7), color=FIG_MID, wrap=True)

    for ext in FIG_FORMATS:
        ext_dir = Path(fig_dir) if flat else Path(fig_dir) / ext
        ext_dir.mkdir(parents=True, exist_ok=True)
        path = ext_dir / f'{name}.{ext}'
        # LZW on the TIFF: uncompressed at 1000 dpi this plate's white space
        # alone runs to hundreds of MB; LZW is lossless and ~1-2% of that size.
        extra = {'pil_kwargs': {'compression': 'tiff_lzw'}} if ext == 'tiff' else {}
        fig.savefig(path, dpi=900, bbox_inches='tight', facecolor='white',
                   **extra)
        print(f'Saved: {path}')
    plt.close(fig)


def with_pooled_row(wide, band_df):
    """Append the pooled-over-conditions row to a per-band counts table."""
    pooled = pooled_table(band_df)
    out = wide.copy()
    out.loc[POOLED_ROW_LABEL] = pooled.iloc[0]
    return out


def _rows_from_wide(wide):
    """[(row_label, [cell, ...]), ...] from a wide (side, timepoint) table."""
    return [(idx, [str(v) for v in wide.loc[idx].tolist()]) for idx in wide.index]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bands', nargs='+', default=DEFAULT_BANDS,
                    help=f'bands to report (default: {" ".join(DEFAULT_BANDS)})')
    ap.add_argument('--out-dir', default=str(OUT_DIR), help='output directory')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    all_sessions = load_all_sessions()
    df = build_long_df(all_sessions, args.bands)

    csvs = {'band_peak_detection_long': df}

    print(f'Bands: {", ".join(args.bands)}   conditions: {len(CONDITIONS)}   '
          f'patients: {NB_PATIENTS}\n')

    tbl = {}
    for band in args.bands:
        sub = df[df.band == band]
        label = BAND_LABELS.get(band, band)
        tbl[band] = with_pooled_row(counts_table(sub), sub)
        print(f'══ {band.upper()} ({label}) ══════════════════════════════════')
        print(_fmt(tbl[band]).to_string(index=False), '\n')
        csvs[f'counts_{band}'] = _fmt(tbl[band])

    print(f'══ Pooled over the {len(CONDITIONS)} conditions ═══════════════════')
    pooled = pooled_table(df)
    print(_fmt(pooled).to_string(index=False), '\n')
    csvs['pooled'] = _fmt(pooled)

    print('══ Patients with a usable stim channel ════════════════════════════')
    avail = availability_table(df)
    print(_fmt(avail).to_string(index=False), '\n')
    csvs['availability'] = _fmt(avail)

    miss = missing_report(df)
    if not miss.empty:
        print('══ Missing patients ═══════════════════════════════════════════')
        print(miss.to_string(index=False), '\n')
        csvs['missing_patients'] = miss

    for name, frame in csvs.items():
        path = out_dir / f'{name}.csv'
        frame.to_csv(path, index=False)
        print(f'Saved: {path}')
    print()

    # ── Article figures ──────────────────────────────────────────────────────
    col_labels   = list(TIMEPOINTS) * len(SIDES)
    n_tp         = len(TIMEPOINTS)
    group_labels = [(f'{side} hemisphere', i * n_tp, (i + 1) * n_tp - 1)
                    for i, side in enumerate(SIDES)]
    band_title   = {b: f'{b.capitalize()} ({BAND_LABELS.get(b, b)})' for b in args.bands}

    n_note = ('Denominators are the patients with a usable stimulation contact at '
              'that timepoint: 10 at M3; 8 (stim-OFF) or 9 (stim-ON) at home; 9 at '
              'M6. P03 has no data after M3; P04 has no at-home stim-OFF recordings.')

    # One figure per band
    for band in args.bands:
        render_table_figure(
            sections=[(None, _rows_from_wide(tbl[band]))],
            col_labels=col_labels, group_labels=group_labels,
            fig_dir=fig_dir, name=f'table_peak_detection_{band}',
            title=f'{band_title[band]} peak detection — patients with a detected '
                  f'peak / n (%)',
            footnote=f'Number of patients with a FOOOF peak detected in the {band} '
                     f'band on the stimulation contact of the given hemisphere, out '
                     f'of the patients with a usable contact, with the detection rate '
                     f'in brackets. The last row pools the {len(CONDITIONS)} '
                     f'conditions (detections / observations). {n_note}',
            midrule_before=(POOLED_ROW_LABEL,),
        )


if __name__ == '__main__':
    main()
