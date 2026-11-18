"""
Wilcoxon signed-rank test: monopolar impedance M3 vs M6.
Two levels, each FDR-corrected (Benjamini-Hochberg) within its own family:
  1. Mean across contacts per patient × hemisphere  (n=9;  2 tests)
  2. Per contact × hemisphere                       (16 tests)

Effect size is the matched-pairs rank-biserial correlation

    r_rb = (W+ - W-) / (n_eff(n_eff + 1)/2)

which is signed (positive -> M6 > M3) and valid under both the exact and the
normal-approximation null.
The earlier Z/sqrt(n) back-derived from the two-sided p-value was unsigned and
inconsistent with the exact null scipy uses at these sample sizes.

Multiplicity correction runs on the unrounded p-values; rounding happens only
on the way out, so tied-by-rounding p-values cannot perturb the BH ranking.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests

DATA_DIR = Path(__file__).parent.parent / "Results" / "Tables"
OUT_DIR  = Path(__file__).parent.parent / "Results" / "stats" / "wilcoxon_impedance"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONTACT_ORDER = ["0", "1a", "1b", "1c", "2a", "2b", "2c", "3"]

ALPHA = 0.05


def rank_biserial(diffs):
    """
    Matched-pairs rank-biserial correlation for the signed-rank test.

        r_rb = (W+ - W-) / (W+ + W-),    W+ + W- = n_eff(n_eff + 1)/2

    Zero differences are dropped (the `zero_method='wilcox'` convention used by
    the test itself), so n_eff can be below the number of pairs.

    Returns (r_rb, n_eff).  r_rb is nan when every difference is zero.
    """
    d  = np.asarray(diffs, dtype=float)
    nz = d[d != 0]
    n_eff = len(nz)
    if n_eff == 0:
        return np.nan, 0

    ranks   = stats.rankdata(np.abs(nz))
    w_plus  = float(ranks[nz > 0].sum())
    w_minus = float(ranks[nz < 0].sum())
    total   = n_eff * (n_eff + 1) / 2.0
    return (w_plus - w_minus) / total, n_eff


def wilcoxon_method(diffs):
    """
    'exact' when the exact null distribution is valid (no zero differences and
    no ties among |diffs|), else 'approx'.  scipy's 'auto' makes the same call,
    but naming it explicitly lets the choice be recorded per row.
    """
    d  = np.asarray(diffs, dtype=float)
    nz = d[d != 0]
    if len(nz) < len(d):
        return "approx"
    abs_nz = np.abs(nz)
    if len(np.unique(abs_nz)) < len(abs_nz):
        return "approx"
    return "exact"


def run_wilcoxon(x, y, label):
    """Run paired Wilcoxon on arrays x (M3) and y (M6). Return result dict."""
    pairs = [(a, b) for a, b in zip(x, y) if not (np.isnan(a) or np.isnan(b))]
    if len(pairs) < 3:
        return None
    x_arr = np.array([p[0] for p in pairs])
    y_arr = np.array([p[1] for p in pairs])
    n = len(pairs)
    diff = y_arr - x_arr                      # positive -> M6 > M3
    r_rb, n_eff = rank_biserial(diff)
    method = wilcoxon_method(diff)

    if n_eff == 0:
        W, p = np.nan, np.nan
    else:
        result = stats.wilcoxon(
            y_arr, x_arr,                     # order matches diff = M6 - M3
            zero_method="wilcox",
            correction=(method == "approx"),
            alternative="two-sided",
            method=method,
        )
        W, p = float(result.statistic), float(result.pvalue)

    return {
        "label":       label,
        "n":           n,
        "n_eff":       n_eff,
        "method":      method,
        "median_M3":   round(np.median(x_arr), 1),
        "median_M6":   round(np.median(y_arr), 1),
        "median_diff": round(np.median(diff), 1),
        "mean_M3":     round(float(np.mean(x_arr)), 1),
        "mean_M6":     round(float(np.mean(y_arr)), 1),
        "mean_diff":   round(float(np.mean(diff)), 1),
        "W":           W,
        "p_raw":       p,                     # unrounded — BH corrects on this
        "r_rb":        round(r_rb, 3) if not np.isnan(r_rb) else np.nan,
    }


def add_fdr(df):
    """BH-correct `p_raw` in place, add rounded p / p_fdr / sig / sig_fdr."""
    p_raw = df["p_raw"].values.astype(float)
    valid = ~np.isnan(p_raw)
    p_fdr = np.full_like(p_raw, np.nan)
    if valid.any():
        _, p_fdr[valid], _, _ = multipletests(p_raw[valid], method="fdr_bh")
    df["p"]        = np.round(p_raw, 4)
    df["p_fdr"]    = np.round(p_fdr, 4)
    df["sig"]      = p_raw < ALPHA
    df["sig_fdr"]  = p_fdr < ALPHA
    return df.drop(columns="p_raw")


# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_DIR / "impedance_all_patients.csv")
mono = df[df["Mode"] == "Monopolar"].copy()
mono["Contact"] = mono["Electrode2"].str.replace(r"(?i)sensight_", "", regex=True)

# Average across sessions per Patient × Timepoint × Hemisphere × Contact
mean_df = (
    mono.groupby(["Patient", "Timepoint", "Hemisphere", "Contact"])["Impedance_Ohm"]
    .mean().reset_index()
)
mean_df = mean_df[mean_df["Timepoint"].isin(["M3", "M6"])]

# ── Level 1: Mean across contacts per patient × hemisphere ─────────────────────
mean_hemi = (
    mono.groupby(["Patient", "Timepoint", "Hemisphere"])["Impedance_Ohm"]
    .mean().reset_index()
)
mean_hemi = mean_hemi[mean_hemi["Timepoint"].isin(["M3", "M6"])]

wide_hemi = mean_hemi.pivot_table(
    index=["Patient", "Hemisphere"], columns="Timepoint", values="Impedance_Ohm"
).reset_index().dropna(subset=["M3", "M6"])

rows_hemi = []
for hemi in ["Left", "Right"]:
    sub = wide_hemi[wide_hemi["Hemisphere"] == hemi]
    res = run_wilcoxon(sub["M3"].values, sub["M6"].values, f"{hemi}")
    if res:
        rows_hemi.append(res)

# FDR across the 2 hemisphere tests (own family, separate from the contact tests)
df_hemi = add_fdr(pd.DataFrame(rows_hemi))

# ── Level 2: Per contact × hemisphere ─────────────────────────────────────────
wide_contact = mean_df.pivot_table(
    index=["Patient", "Hemisphere", "Contact"],
    columns="Timepoint", values="Impedance_Ohm"
).reset_index().dropna(subset=["M3", "M6"])

rows_contact = []
for hemi in ["Left", "Right"]:
    for contact in CONTACT_ORDER:
        sub = wide_contact[
            (wide_contact["Hemisphere"] == hemi) &
            (wide_contact["Contact"] == contact)
        ]
        res = run_wilcoxon(sub["M3"].values, sub["M6"].values, f"{hemi} – {contact}")
        if res:
            res["Hemisphere"] = hemi
            res["Contact"]    = contact
            rows_contact.append(res)

# FDR across all contact tests (both hemispheres pooled into one family)
df_contact = add_fdr(pd.DataFrame(rows_contact))

# ── Save ──────────────────────────────────────────────────────────────────────
out = OUT_DIR / "wilcoxon_impedance_M3_M6.xlsx"
with pd.ExcelWriter(out, engine="openpyxl") as writer:
    df_hemi.to_excel(writer,    sheet_name="Mean_per_Hemisphere", index=False)
    df_contact.to_excel(writer, sheet_name="Per_Contact_FDR",     index=False)

print(f"Saved: {out}\n")

COLS = ["label", "n", "n_eff", "method",
        "median_M3", "median_M6", "median_diff",
        "mean_M3", "mean_M6", "mean_diff",
        "W", "p", "p_fdr", "sig", "sig_fdr", "r_rb"]

print("── Mean per hemisphere (FDR across 2 tests) ─────────")
print(df_hemi[COLS].to_string(index=False))

print("\n── Per contact (FDR across 16 tests) ────────────────")
print(df_contact[COLS].to_string(index=False))
