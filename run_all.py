#!/usr/bin/env python3
"""
run_all.py — one entry point that regenerates the whole ECOLOVIM analysis.

Every step of the pipeline is declared below as a `Stage`: the command that
produces it, the files it reads, the files it writes, and the source files whose
content it depends on.  The runner walks that graph in dependency order and
re-runs a stage only when something it depends on actually changed.

    change config.py     ->  features rebuild -> tremor -> every stats module
                             -> every table and figure downstream of them

Staleness is decided per stage from three things:
    1. any declared output missing                       -> stale
    2. fingerprint changed (command string, content hash of the code files,
       size+mtime of the data inputs)                    -> stale
    3. an upstream stage re-ran in this session          -> stale

State lives in Results/.run_all_state.json, per-stage logs in Results/logs/.

Usage
-----
    python run_all.py                     # rebuild whatever is stale
    python run_all.py --list              # show the graph + what is stale
    python run_all.py --dry-run           # say what would run, run nothing
    python run_all.py --force             # rebuild everything selected
    python run_all.py --only lme icc      # those stages only
    python run_all.py --from tremor-bands # that stage and all its descendants
    python run_all.py --group figures     # one group
    python run_all.py --skip features-M3 features-M6
    python run_all.py -j 4                # run independent stages in parallel
    python run_all.py -v                  # stream stage output to the console

Exit code is 0 only if every selected stage ended `ok` or `skip`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATE_FILE = ROOT / 'Results' / '.run_all_state.json'
LOG_DIR = ROOT / 'Results' / 'logs'
PY = sys.executable

# ══════════════════════════════════════════════════════════════════════════════
# Stage declarations
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Stage:
    name: str
    group: str
    cmd: list[str]
    outputs: tuple[str, ...]                       # glob patterns, relative to ROOT
    code: tuple[str, ...] = ()                     # source files -> content hashed
    inputs: tuple[str, ...] = ()                   # data files -> size + mtime
    needs: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    note: str = ''

def _mod(name: str) -> list[str]:
    return [PY, '-m', name]

def _script(name: str, *args: str) -> list[str]:
    return [PY, name, *args]

# Source files that essentially everything downstream of the features depends on.
_CORE = ('config.py',)
_STATS_CORE = _CORE + ('stats_analysis/data_utils.py', 'stats_analysis/plot_style.py')
_FIG_CORE = _STATS_CORE + ('table_figure_style.py',)

_FEATURE_PKLS = (
    'Results/features/spect_feats_dict_M3_badremoved.pkl',
    'Results/features/spect_feats_dict_M6_badremoved.pkl',
    'Results/features/home_psds_dict.pkl',
)
# The tremor keys now live inside _FEATURE_PKLS: pipeline.py calls
# add_tremor_bands() before saving, so no *_tremor.pkl copy is produced.
# Kept empty so the stage definitions below need no edit.
_TREMOR_PKLS = ()

# pipeline.py writes the tremor keys into the feature pickles itself (it calls
# add_tremor_bands before dumping), so the whole tremor chain is a code
# dependency OF THE FEATURES, not only of the tremor-bands stage that reports on
# them.  The arrow in the graph points the other way — tremor-bands needs
# features — which is why these three were missed:
#
#   pipeline.py -> add_tremor_bands.py -> tremor_bands.py         (TREMOR_BANDS)
#                                      -> stats_analysis/data_utils.py
#                                         (STIM_CHANNELS, CONDITIONS, EXCLUDED_PATIENTS)
#
# Without them, editing a patient's tremor window or a stimulation contact left
# features-M3/M6 reading "up to date" with the old numbers baked into the
# pickles, and since the pickle mtime never moved, every stats stage downstream
# stayed green too — while tremor-bands re-ran and refreshed the QC report
# describing pickles that had not changed.
#
# pipeline.py runs one of two halves per invocation (`if session in
# CLINIC_SESSIONS: ... else: ...`), so the code each half depends on is split
# here too: editing a clinic-only file does not rebuild the at-home pickle, and
# editing power.py's at-home band power does not rebuild the clinic sessions.
_PIPELINE_SHARED = (
    'pipeline.py',
    'add_tremor_bands.py', 'tremor_bands.py',      # both halves add tremor keys
    'spectral_analysis/frequency_analysis.py',
    'stats_analysis/data_utils.py',                # STIM_CHANNELS, CONDITIONS
) + _CORE

# Clinic only: Welch/FOOOF over the preprocessed epochs.
_CLINIC_CODE = _PIPELINE_SHARED + ('spectral_analysis/get_spectral_feat.py',)

# Home only: the device's stored FFT bins, band power and FOOOF over them.
_HOME_CODE = _PIPELINE_SHARED + ('power.py',)

STAGES: list[Stage] = [

    # ── features ──────────────────────────────────────────────────────────────
    # One stage per timepoint, matching data_utils.TIMEPOINTS and the if/else in
    # pipeline.py.  $ECOLOVIM_SESSION picks the branch; each stage owns exactly
    # the files its branch writes, so the three can run concurrently.
    Stage(
        name='features-M3', group='features',
        cmd=_script('pipeline.py'),
        env={'ECOLOVIM_SESSION': 'M3'},
        code=_CLINIC_CODE,
        inputs=('Results/data-preprocessed/epochs_dict_M3_badremoved.pkl',),
        outputs=('Results/features/spect_feats_dict_M3_badremoved.pkl',),
        note='~6s: preprocessed 800 ms epochs -> Welch PSD + FOOOF',
    ),
    Stage(
        name='features-M6', group='features',
        cmd=_script('pipeline.py'),
        env={'ECOLOVIM_SESSION': 'M6'},
        code=_CLINIC_CODE,
        inputs=('Results/data-preprocessed/epochs_dict_M6_badremoved.pkl',),
        outputs=('Results/features/spect_feats_dict_M6_badremoved.pkl',),
        note='~5s, same for M6',
    ),
    Stage(
        name='features-home', group='features',
        cmd=_script('pipeline.py'),
        env={'ECOLOVIM_SESSION': 'home'},
        code=_HOME_CODE,
        inputs=('Results/data-preprocessed/home_fft_dict.pkl',),
        outputs=('Results/features/home_psds_dict.pkl',),
        note='~1s: the device computed these spectra on-chip, so this branch only '
             'reads the stored FFT bins, sums band power over them, fits FOOOF and '
             'adds the tremor keys.  No epoching, no Welch',
    ),

    # ── tremor bands: re-sums stored PSDs over the per-patient tremor windows ──
    Stage(
        name='tremor-bands', group='features',
        cmd=_script('add_tremor_bands.py'),
        needs=('features-M3', 'features-M6', 'features-home'),
        code=('add_tremor_bands.py', 'tremor_bands.py') + _CORE,
        inputs=_FEATURE_PKLS,
        outputs=('Results/stats/tremor/tremor_band_bins.csv',),
    ),

    # ── statistics ────────────────────────────────────────────────────────────
    Stage(
        name='lme', group='stats',
        cmd=_mod('stats_analysis.lme_analysis'),
        needs=('tremor-bands',),
        code=('stats_analysis/lme_analysis.py',) + _STATS_CORE,
        inputs=_FEATURE_PKLS + _TREMOR_PKLS,
        # The per-condition .txt fits and the leave-one-out CSVs are outputs
        # too, so a missing one marks the stage stale.
        outputs=('Results/stats/lme/lme_m3m6_summary_L.csv',
                 'Results/stats/lme/lme_m3m6_summary_R.csv',
                 'Results/stats/lme/lme_m3m6_summary_bilateral.csv',
                 'Results/stats/lme/lme_percond_*.txt',
                 'Results/stats/lme/loo_percond_*.csv',
                 'Results/stats/lme/table_percond_all_*.html'),
        note=('~5.5 h (measured 140 min at 1000 permutations, scaled by refit '
              'count — not re-measured): every reported p is a permutation p and '
              'every interval a cluster bootstrap, 5000 + 2000 refits per cell. '
              'ECOLOVIM_LME_JOBS=8 spreads the 52 cells per unit over processes: '
              'cells are independent and self-seeded, so the CSVs are '
              'bit-identical (verified).  Measured 1.7x on 4 workers at a small '
              'budget, where per-worker import cost still dominates; at the full '
              'budget (~76 s per cell) that cost amortises and 8 workers should '
              'land near ~1 h, but that has NOT been measured.  It is not set here on purpose — a '
              'stage env var goes into the fingerprint, and a core count must not '
              'force a rebuild.  Fast preview: --only lme --args ""  with '
              'ECOLOVIM_LME_NPERM=200 ECOLOVIM_LME_NBOOT=300 in the environment '
              '(a preview, not a result)'),
    ),
    Stage(
        name='sem', group='stats',
        cmd=_mod('stats_analysis.sem_analysis'),
        needs=('tremor-bands',),
        code=('stats_analysis/sem_analysis.py',) + _STATS_CORE,
        inputs=_FEATURE_PKLS + _TREMOR_PKLS,
        outputs=('Results/stats/icc/sem_all_sides.csv',),
        note='SEM / MDC variance components per (side, feature, condition, pair). '
             'Upstream of icc: icc_analysis outlines cells with it in the ICC '
             'tables',
    ),
    Stage(
        name='icc', group='stats',
        cmd=_mod('stats_analysis.icc_analysis'),
        needs=('tremor-bands', 'sem'),
        code=('stats_analysis/icc_analysis.py', 'stats_analysis/sem_analysis.py') + _STATS_CORE,
        inputs=_FEATURE_PKLS + _TREMOR_PKLS + ('Results/stats/icc/sem_all_sides.csv',),
        outputs=('Results/stats/icc/icc_all_pairwise_L.csv',
                 'Results/stats/icc/icc_all_pairwise_R.csv',
                 'Results/stats/icc/icc_all_pairwise_bilateral.csv',
                 'Results/stats/icc/icc_table_*.html',
                 'Results/stats/icc/icc_scatter_*.html'),
        note='~35s including the bootstrap CIs',
    ),
    Stage(
        name='lme-rest-mvt', group='stats',
        cmd=_mod('stats_analysis.lme_rest_mvt_diff'),
        needs=('tremor-bands',),
        code=('stats_analysis/lme_rest_mvt_diff.py',) + _STATS_CORE,
        inputs=_FEATURE_PKLS + _TREMOR_PKLS,
        outputs=('Results/stats/lme_rest_mvt/lme_rest_mvt_diff_L.csv',
                 'Results/stats/lme_rest_mvt/lme_rest_mvt_diff_R.csv',
                 'Results/stats/lme_rest_mvt/lme_rest_mvt_diff_bilateral.csv',
                 'Results/stats/lme_rest_mvt/diff_long_L.csv',
                 'Results/stats/lme_rest_mvt/diff_long_R.csv',
                 'Results/stats/lme_rest_mvt/diff_long_bilateral.csv',
                 'Results/stats/lme_rest_mvt/simple_effects_L.csv',
                 'Results/stats/lme_rest_mvt/simple_effects_R.csv',
                 'Results/stats/lme_rest_mvt/simple_effects_bilateral.csv'),
        note='THE slow one: ~4.5 h (measured 100 min at 1000 permutations, scaled by refit count — not re-measured; label permutation + cluster bootstrap, plus since fix A a per-state reduced-model bootstrap for the simple effects, ~+15 min).  --args --jobs 8 spreads the 13 features per side over processes; bit-identical (verified), measured 2.0x on 4 workers at a small budget, ~1 h at the full budget is an extrapolation and not measured.  For a fast pass: --only lme-rest-mvt --args --no-perm --no-boot',
    ),
    Stage(
        name='peak-params', group='stats',
        cmd=_mod('stats_analysis.peak_params_analysis'),
        needs=('tremor-bands',),
        code=('stats_analysis/peak_params_analysis.py',) + _STATS_CORE,
        inputs=_FEATURE_PKLS,
        outputs=('Results/stats/peak_params/peak_params_availability.csv',
                 'Results/stats/peak_params/peak_params_long.csv'),
        note='detection/availability only, no models',
    ),
    Stage(
        name='band-peaks', group='stats',
        cmd=_mod('stats_analysis.band_peak_detection'),
        needs=('tremor-bands',),
        code=('stats_analysis/band_peak_detection.py',) + _STATS_CORE,
        inputs=_FEATURE_PKLS,
        outputs=('Results/stats/band_peaks/*.csv',),
    ),

    # ── impedance chain ───────────────────────────────────────────────────────
    # impedance_all_patients.csv comes from the raw session JSONs, which are not
    # distributed.  make_synthetic_data.py writes a fake CSV at the same path.
    Stage(
        name='impedance-wilcoxon', group='impedance',
        cmd=_script('stats_analysis/wilcoxon_impedance.py'),
        code=('stats_analysis/wilcoxon_impedance.py',) + _CORE,
        inputs=('Results/Tables/impedance_all_patients.csv',),
        outputs=('Results/stats/wilcoxon_impedance/wilcoxon_impedance_M3_M6.xlsx',),
    ),

    # ── figures ───────────────────────────────────────────────────────────────
    Stage(
        name='fig-lme-effectsize', group='figures',
        cmd=_script('make_lme_effectsize_figure.py'),
        needs=('lme',),
        # row labels are imported from the ICC summary plate
        code=('make_lme_effectsize_figure.py', 'make_icc_summary_figure.py') + _FIG_CORE,
        inputs=('Results/stats/lme/lme_m3m6_summary_bilateral.csv',),
        outputs=('Results/Figures/fig_lme_effectsize_bilateral.png',
                 'Results/Figures/fig_lme_effectsize_bilateral.tiff'),
        note='Cohen\'s d plate for the bilateral LME contrasts: all 156 estimates as '
             'dots — area is |d|, hue is the sign on a diverging scale with a neutral '
             'grey zero, and the ring encodes evidence (thick = BH q < 0.05, thin = '
             'nominal p only)',
    ),
    Stage(
        name='fig-icc-summary', group='figures',
        cmd=_script('make_icc_summary_figure.py'),
        needs=('icc',),
        code=('make_icc_summary_figure.py',) + _FIG_CORE,
        inputs=('Results/stats/icc/icc_all_pairwise_L.csv',
                'Results/stats/icc/icc_all_pairwise_R.csv',
                'Results/stats/icc/icc_all_pairwise_bilateral.csv'),
        outputs=tuple(f'Results/Figures/fig_icc_summary_stim_{side}.{ext}'
                      for side in ('L', 'R', 'bilateral')
                      for ext in ('png', 'tiff')),
        note='narrative companion to fig-icc-pairwise: one panel per stimulation '
             'state, two segments per feature.  Same row order and labels as the '
             'heatmap so they read side by side with it',
    ),
    Stage(
        name='fig-rest-mvt-combined', group='figures',
        cmd=_script('make_rest_mvt_combined_figure.py'),
        needs=('lme-rest-mvt', 'tremor-bands'),
        # the psd/forest modules are imported, not run: they are code deps here
        code=('make_rest_mvt_combined_figure.py',
              'make_rest_mvt_psd_figure.py',
              'make_rest_mvt_forest_figure.py') + _FIG_CORE,
        inputs=('Results/stats/lme_rest_mvt/simple_effects_bilateral.csv',
                'Results/stats/lme_rest_mvt/lme_rest_mvt_diff_bilateral.csv') + _FEATURE_PKLS,
        outputs=('Results/Figures/fig_rest_mvt_panels_bilateral.png',
                 'Results/Figures/fig_rest_mvt_panels_bilateral.tiff'),
        note='~5s once panel A\'s cluster bootstrap is cached; --args --refresh-boot recomputes it and is much slower',
    ),
    # Same figure for each hemisphere: the script draws one side per call.
    *(Stage(
        name=f'fig-rest-mvt-combined-{side}', group='figures',
        cmd=_script('make_rest_mvt_combined_figure.py', '--side', side),
        needs=('lme-rest-mvt', 'tremor-bands'),
        code=('make_rest_mvt_combined_figure.py',
              'make_rest_mvt_psd_figure.py',
              'make_rest_mvt_forest_figure.py') + _FIG_CORE,
        inputs=(f'Results/stats/lme_rest_mvt/simple_effects_{side}.csv',
                f'Results/stats/lme_rest_mvt/lme_rest_mvt_diff_{side}.csv') + _FEATURE_PKLS,
        outputs=(f'Results/Figures/fig_rest_mvt_panels_{side}.png',) + extra,
        note=f'{side} companion to fig-rest-mvt-combined',
    ) for side, extra in (('L', ('Results/Figures/fig_rest_mvt_panels_L.tiff',)),
                          ('R', ()))),
    Stage(
        name='fig-trajectory', group='figures',
        cmd=_script('make_trajectory_figure.py'),
        needs=('tremor-bands',),
        code=('make_trajectory_figure.py',) + _FIG_CORE,
        inputs=_FEATURE_PKLS + _TREMOR_PKLS,
        # draw() is called once for the headline feature set, then once per
        # SUPP_PAGE_SIZES chunk of the 13-feature supplement.
        outputs=tuple(
            f'Results/Figures/fig_trajectories{stem}.png'
            for side in ('bilateral', 'L', 'R')
            for stem in (f'_{side}', f'_all_{side}_1', f'_all_{side}_2')
        ) + ('Results/Figures/fig_trajectories_bilateral.tiff',),
    ),
    Stage(
        name='fig-impedance', group='figures',
        cmd=_script('make_impedance_figure.py'),
        needs=('impedance-wilcoxon',),
        code=('make_impedance_figure.py',) + _FIG_CORE,
        inputs=('Results/Tables/impedance_all_patients.csv',
                'Results/stats/wilcoxon_impedance/wilcoxon_impedance_M3_M6.xlsx'),
        outputs=('Results/Figures/impedance_per_contact_grid.png',),
    ),
]

BY_NAME = {s.name: s for s in STAGES}

# ══════════════════════════════════════════════════════════════════════════════
# Graph helpers
# ══════════════════════════════════════════════════════════════════════════════

def topo_order(stages: list[Stage]) -> list[Stage]:
    """Dependency order; raises on an unknown dep or a cycle."""
    for s in stages:
        for dep in s.needs:
            if dep not in BY_NAME:
                raise SystemExit(f'stage {s.name!r} needs unknown stage {dep!r}')
    done: set[str] = set()
    out: list[Stage] = []
    remaining = list(stages)
    names = {s.name for s in stages}
    while remaining:
        ready = [s for s in remaining if all(d in done or d not in names for d in s.needs)]
        if not ready:
            stuck = ', '.join(s.name for s in remaining)
            raise SystemExit(f'dependency cycle among: {stuck}')
        ready.sort(key=lambda s: STAGES.index(s))
        for s in ready:
            out.append(s)
            done.add(s.name)
        remaining = [s for s in remaining if s.name not in done]
    return out

def descendants(names: set[str]) -> set[str]:
    """`names` plus every stage that transitively depends on them."""
    out = set(names)
    changed = True
    while changed:
        changed = False
        for s in STAGES:
            if s.name not in out and any(d in out for d in s.needs):
                out.add(s.name)
                changed = True
    return out

def levels(stages: list[Stage]) -> list[list[Stage]]:
    """Group stages into waves that may run concurrently."""
    names = {s.name for s in stages}
    depth: dict[str, int] = {}
    for s in topo_order(stages):
        deps = [depth[d] for d in s.needs if d in names]
        depth[s.name] = max(deps, default=-1) + 1
    out: list[list[Stage]] = [[] for _ in range(max(depth.values(), default=-1) + 1)]
    for s in stages:
        out[depth[s.name]].append(s)
    return out

# ══════════════════════════════════════════════════════════════════════════════
# Fingerprints
# ══════════════════════════════════════════════════════════════════════════════

def expand(patterns: tuple[str, ...]) -> list[Path]:
    found: list[Path] = []
    for pat in patterns:
        if any(ch in pat for ch in '*?['):
            found.extend(sorted(ROOT.glob(pat)))
        else:
            found.append(ROOT / pat)
    return found

def fingerprint(stage: Stage) -> str:
    """
    Hash of everything that should force a rebuild: the command, the *content*
    of the stage's source files, and the size+mtime of its data inputs (hashing
    30 MB pickles on every check would be slower than the analysis).
    """
    h = hashlib.sha1()
    h.update(' '.join(stage.cmd[1:]).encode())          # skip the interpreter path
    h.update(repr(sorted(stage.env.items())).encode())

    for p in expand(stage.code):
        h.update(str(p.relative_to(ROOT)).encode())
        h.update(p.read_bytes() if p.is_file() else b'<missing>')

    for p in expand(stage.inputs):
        h.update(str(p.relative_to(ROOT)).encode())
        if p.is_file():
            st = p.stat()
            h.update(f'{st.st_size}:{st.st_mtime_ns}'.encode())
        else:
            h.update(b'<missing>')
    return h.hexdigest()

def missing_outputs(stage: Stage) -> list[str]:
    out = []
    for pat in stage.outputs:
        if any(ch in pat for ch in '*?['):
            if not list(ROOT.glob(pat)):
                out.append(pat)
        elif not (ROOT / pat).exists():
            out.append(pat)
    return out

def why_stale(stage: Stage, state: dict, dirty: set[str]) -> str:
    up = [d for d in stage.needs if d in dirty]
    if up:
        return f'upstream re-ran ({", ".join(up)})'
    miss = missing_outputs(stage)
    if miss:
        return f'output missing ({miss[0]}{"…" if len(miss) > 1 else ""})'
    prev = state.get(stage.name, {})
    if not prev:
        return 'never recorded'
    if prev.get('fingerprint') != fingerprint(stage):
        return 'inputs or code changed'
    return ''

# ══════════════════════════════════════════════════════════════════════════════
# Execution
# ══════════════════════════════════════════════════════════════════════════════

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except json.JSONDecodeError:
            print('! state file unreadable, treating everything as stale')
    return {}

def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True))

def run_stage(stage: Stage, verbose: bool, extra_args: list[str]) -> tuple[bool, float, Path]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / f'{stage.name}.log'
    env = {**os.environ, 'MPLBACKEND': 'Agg', 'PYTHONUNBUFFERED': '1', **stage.env}
    cmd = stage.cmd + extra_args
    t0 = time.time()
    with log.open('w') as fh:
        fh.write(f'# {datetime.now().isoformat(timespec="seconds")}\n'
                 f'# cwd={ROOT}\n# cmd={" ".join(cmd)}\n'
                 f'# env={stage.env}\n\n')
        fh.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:                       # type: ignore[union-attr]
            fh.write(line)
            if verbose:
                sys.stdout.write(f'[{stage.name}] {line}')
        rc = proc.wait()
    return rc == 0, time.time() - t0, log

def tail(path: Path, n: int = 30) -> str:
    try:
        lines = path.read_text(errors='replace').splitlines()
    except OSError:
        return '(no log)'
    return '\n'.join(f'    | {ln}' for ln in lines[-n:])

def fmt_dur(sec: float) -> str:
    if sec < 90:
        return f'{sec:.0f}s'
    return f'{sec / 60:.1f}min'

# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def select(args: argparse.Namespace) -> list[Stage]:
    if args.only:
        unknown = [n for n in args.only if n not in BY_NAME]
        if unknown:
            raise SystemExit(f'unknown stage(s): {", ".join(unknown)}')
        chosen = set(args.only)
    elif args.start_from:
        if args.start_from not in BY_NAME:
            raise SystemExit(f'unknown stage: {args.start_from}')
        chosen = descendants({args.start_from})
    elif args.group:
        chosen = {s.name for s in STAGES if s.group in args.group}
    else:
        chosen = {s.name for s in STAGES if s.enabled}

    if not args.only:                                   # --only is explicit intent
        chosen -= {s.name for s in STAGES if not s.enabled}
    chosen -= set(args.skip or [])
    if not chosen:
        raise SystemExit('nothing selected')
    return topo_order([s for s in STAGES if s.name in chosen])

def cmd_list(state: dict) -> None:
    print(f'{"stage":<24} {"group":<10} {"status":<22} deps')
    print('─' * 100)
    for s in topo_order(STAGES):
        if not s.enabled:
            status = 'disabled'
        else:
            status = why_stale(s, state, set()) or 'up to date'
        deps = ', '.join(s.needs) or '—'
        print(f'{s.name:<24} {s.group:<10} {status[:22]:<22} {deps}')
        if s.note:
            print(f'{"":<24} {"":<10} {"":<22} ↳ {s.note}')

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--list', action='store_true', help='print the graph and exit')
    ap.add_argument('--adopt', action='store_true',
                    help='baseline the Results tree as it stands: record the current '
                         'fingerprint of every selected stage whose outputs already '
                         'exist, without running anything.  Run this once so the '
                         'first real run only rebuilds what you actually changed.')
    ap.add_argument('--only', nargs='+', metavar='STAGE',
                    help='run exactly these stages (dependencies are not pulled in)')
    ap.add_argument('--from', dest='start_from', metavar='STAGE',
                    help='run this stage and everything downstream of it')
    ap.add_argument('--group', nargs='+',
                    choices=sorted({s.group for s in STAGES}),
                    help='run one or more groups')
    ap.add_argument('--skip', nargs='+', metavar='STAGE', help='exclude these stages')
    ap.add_argument('--force', action='store_true',
                    help='rebuild the selected stages even if they look current')
    ap.add_argument('--dry-run', action='store_true', help='say what would run')
    ap.add_argument('--keep-going', action='store_true',
                    help='continue after a stage fails instead of stopping')
    ap.add_argument('-j', '--jobs', type=int, default=1,
                    help='stages to run concurrently within a dependency level')
    ap.add_argument('-v', '--verbose', action='store_true',
                    help='stream stage output to the console as well as the log')
    ap.add_argument('--args', nargs=argparse.REMAINDER, default=[],
                    help='pass the remaining arguments through to every stage '
                         '(only sensible with --only)')
    args = ap.parse_args(argv)

    state = load_state()

    if args.list:
        cmd_list(state)
        return 0

    stages = select(args)

    if args.adopt:
        adopted = skipped = 0
        for s_ in stages:
            miss = missing_outputs(s_)
            if miss:
                print(f'  --    {s_.name:<24} not adopted, output missing ({miss[0]})')
                skipped += 1
                continue
            state[s_.name] = {
                'fingerprint': fingerprint(s_),
                'finished': 'adopted ' + datetime.now().isoformat(timespec='seconds'),
                'duration_s': 0.0,
            }
            adopted += 1
            print(f'  adopt {s_.name:<24} marked up to date')
        save_state(state)
        print(f'\n{adopted} adopted · {skipped} still stale')
        return 0

    dirty: set[str] = set()
    results: list[tuple[str, str, float]] = []
    failed = False

    print(f'ECOLOVIM · {len(stages)} stage(s) selected · logs in Results/logs/')
    print('─' * 72)

    for wave in levels(stages):
        todo = []
        for s in wave:
            reason = 'forced' if args.force else why_stale(s, state, dirty)
            if not reason:
                results.append((s.name, 'skip', 0.0))
                print(f'  skip  {s.name:<24} up to date')
                continue
            todo.append((s, reason))

        if args.dry_run:
            for s, reason in todo:
                results.append((s.name, 'would-run', 0.0))
                print(f'  run   {s.name:<24} {reason}')
                dirty.add(s.name)
            continue

        def _work(item):
            s, reason = item
            print(f'  run   {s.name:<24} {reason}')
            ok, dur, log = run_stage(s, args.verbose, args.args)
            return s, ok, dur, log

        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            for s, ok, dur, log in pool.map(_work, todo):
                if ok:
                    state[s.name] = {
                        'fingerprint': fingerprint(s),          # recompute: outputs moved
                        'finished': datetime.now().isoformat(timespec='seconds'),
                        'duration_s': round(dur, 1),
                    }
                    save_state(state)
                    dirty.add(s.name)
                    results.append((s.name, 'ok', dur))
                    print(f'  ok    {s.name:<24} {fmt_dur(dur)}')
                else:
                    state.pop(s.name, None)
                    save_state(state)
                    results.append((s.name, 'FAIL', dur))
                    failed = True
                    print(f'  FAIL  {s.name:<24} {fmt_dur(dur)}  → {log}')
                    print(tail(log))

        if failed and not args.keep_going:
            print('\nstopping (use --keep-going to continue past failures)')
            break

    print('─' * 72)
    n_ok = sum(1 for _, st, _ in results if st == 'ok')
    n_skip = sum(1 for _, st, _ in results if st == 'skip')
    n_fail = sum(1 for _, st, _ in results if st == 'FAIL')
    total = sum(d for _, _, d in results)
    print(f'{n_ok} rebuilt · {n_skip} up to date · {n_fail} failed · {fmt_dur(total)}')
    if n_fail:
        print('failed: ' + ', '.join(n for n, st, _ in results if st == 'FAIL'))
    return 1 if n_fail else 0

if __name__ == '__main__':
    raise SystemExit(main())
