"""diag_trench_pull_smoothing — how much of the trench pull's jaggedness is x_I?

DIAGNOSTIC, not a paper figure — hence diag_ rather than fig_, which is
reserved for the manuscript's one-to-one figure set. Writes
figures/diag_trench_pull_smoothing.png and tables/diag_trench_pull_smoothing.csv.

Requested by Dan 2026-09-23, two questions in one:

  (a) show the trench pull through time under a range of smoothing
      operations, so the size and character of the jaggedness is visible
      rather than asserted from a roughness statistic;
  (b) his push-back that V at the trench "does not look especially
      smooth" in the existing figures.

ON (b) HE IS READING A NORMALISED AXIS. fig_slab_correlations divides
every curve by its own standard deviation so that quantities in different
units can share an axis. That makes a small, smooth wiggle look exactly
as large as a big, smooth one, and it is why V reads as noisy there. Row
4 plots V and Delta N_D in ABSOLUTE TN/m, unsmoothed, on a shared axis.
Their roughness is 0.65 against 0.64 (STD) and 0.74 against 0.62 (WAL) --
V is marginally rougher than Delta N_D and nowhere near the trench pull's
1.07. The normalisation, not the data, is what made it look otherwise.

⚠ THIS FIGURE DOES NOT RE-PICK x_I. The cache stores profiles AT the
picked columns, not the fields, so a genuine re-pick needs a ~20 min
rebuild. What row 3 does instead is smooth the x_I COLUMN PROFILE through
time and recompute

    trench pull = integral of (sigma_zz at x_T  -  sigma_zz at x_I)

from it. That is a faithful PROXY for a temporally stabilised pick -- it
removes exactly the quantity that a stabilised x_I would stop injecting --
but it is not the same thing as a better picker, and the result should be
read as "this is what is available to be recovered", not as a corrected
measurement.

WHAT IT SHOWS.

  1  trench pull under four smoothers, with the snapshots marked. Variance
     removed: 3-point median 24 % (STD) / 25 % (WAL), 5-point
     Savitzky-Golay 25 % / 17 %, Gaussian sigma = 1 step 47 % / 34 %,
     sigma = 2 steps 66 % / 59 %. Note how little the median and the
     Savitzky-Golay differ: the noise is NOT a few isolated excursions of
     the kind that afflict w_T (§5.5), it is present at nearly every
     snapshot. Only a genuinely low-pass operation removes much of it, and
     a sigma = 2 filter is already eating into the 25 Myr signal.
  2  the two columns the trench pull is built from, detrended, in absolute
     TN/m. The trench column is smooth (roughness 0.54 / 0.39) and the x_I
     column is not (1.04 / 0.83); on one axis the asymmetry needs no
     statistic. STD is the clearer case.
  3  raw trench pull against the stabilised-reference proxy. Roughness
     falls 1.07 -> 0.65 (STD) and 0.91 -> 0.56 (WAL) -- to the level of
     Delta N_D (0.65 / 0.60) and V (0.65 / 0.74) -- while the amplitude
     is very nearly untouched: s.d. 0.145 -> 0.126 (87 % retained) and
     0.270 -> 0.265 (98 %). The high-frequency share collapses from 51 %
     to 31 % and from 22 % to 5 %.
  4  V, N_D and the trench pull at the trench in ABSOLUTE units, raw --
     the answer to (b). In STD V is visibly the smallest and smoothest of
     the three; the trench pull is the jagged one.

CONCLUSION. The trench pull's excess roughness is the x_I reference,
and removing it returns a series as smooth as the other balance terms
while keeping 87-98 % of the amplitude. Nothing is being smoothed away
except the reference noise. That is the case for fixing the picker rather
than working around it -- and it also means no amount of post-hoc
smoothing of the trench pull is a substitute, since a filter strong
enough to do the same job (sigma = 2 steps, -66 % variance) also removes
real signal.

DRAFT CAPTION (if it ever needs one). Diagnostic of the trench pull's
short-timescale noise, STD (left) and WAL (right). Row 1: the trench pull
under four smoothing operations. Row 2: the two column integrals it is
formed from, showing that the first-isostatic column rather than the
trench column carries the jaggedness. Row 3: the raw trench pull against
a proxy for a temporally stabilised x_I reference. Row 4: the shear
resultant and the normal-stress-difference resultant at the trench in
absolute units, unsmoothed.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter, gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import column_profiles_cache as cpc
from fig_budget_time import terms
from tables_io import write_table

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C_RULE = '#BFC3D1'
C_RAW = '0.60'
C_TRENCH = '#0072B2'
C_XI = '#D55E00'
C_STAB = '#1B9E77'
C_V = '#7B3294'


def rough(y):
    """sd(step-to-step difference) / sd. White noise = sqrt(2); a clean
    25 Myr signal sampled every 2 Myr is about 0.5."""
    return float(np.diff(y).std() / y.std())


def hf_share(t, y, cut_myr=12.0):
    """Share of variance at periods shorter than cut_myr, on a uniform grid."""
    tu = np.linspace(t[0], t[-1], len(t))
    yu = np.interp(tu, t, y); yu = yu - yu.mean()
    P = np.abs(np.fft.rfft(yu)) ** 2
    f = np.fft.rfftfreq(len(yu), tu[1] - tu[0])
    return float(100 * P[f > 1 / cut_myr].sum() / P[1:].sum())


def main():
    if not hasattr(np, 'trapezoid'):
        np.trapezoid = np.trapz
    d = cpc.load()
    fig, axes = plt.subplots(4, 2, figsize=(13.4, 13.0), sharex='col')
    rows = [('model', 'quantity', 'value')]
    ld = lambda t, y: y - np.polyval(np.polyfit(t, y, 1), t)

    for col, key in enumerate(('STD', 'WAL')):
        c = cpc.derive(d, key)
        q = terms(d, key)
        r = cpc.resultants(d, key)
        z, zc, t = c['z'], c['zc'], c['t']
        I = lambda a: np.trapezoid(a[:, zc], z[zc], axis=1) / 1e12
        TP = -I(c['p_T'])
        szz_T, szz_I = I(d[f'{key}_szz_T']), I(d[f'{key}_szz_I'])

        # --- row 1: the trench pull under four smoothers ----------------
        sm = {
            '3-point running median': median_filter(TP, size=3, mode='nearest'),
            '5-point Savitzky–Golay': savgol_filter(TP, 5, 2),
            'Gaussian $\\sigma$ = 1 step (2 Myr)': gaussian_filter1d(TP, 1.0,
                                                                    mode='nearest'),
            'Gaussian $\\sigma$ = 2 steps (4 Myr)': gaussian_filter1d(TP, 2.0,
                                                                     mode='nearest'),
        }
        a0 = axes[0, col]
        a0.plot(t, TP, '-', color=C_RAW, lw=3.0, alpha=0.65, label='raw')
        a0.plot(t, TP, '.', color='k', ms=3.5, zorder=5)
        for (nm, y), c_ in zip(sm.items(), (C_TRENCH, C_XI, C_STAB, 'k')):
            drop = 100 * (1 - ld(t, y).var() / ld(t, TP).var())
            a0.plot(t, y, '-', color=c_, lw=1.5,
                    label=f'{nm}  (−{drop:.0f} % variance)')
            rows += [(key, f'variance_removed_percent_'
                      f'{nm.split("(")[0].strip().replace(" ", "_").replace("$", "")}',
                      f'{drop:.1f}')]
        a0.set_ylabel('trench pull [TN/m]', fontsize=10)
        a0.set_title(f'{key}   trench pull, raw roughness {rough(ld(t, TP)):.2f}, '
                     f'{hf_share(t, ld(t, TP)):.0f} % of variance below 12 Myr',
                     fontsize=10)
        a0.legend(frameon=False, fontsize=8, loc='upper left', ncol=2)

        # --- row 2: which of the two columns is noisy -------------------
        a1 = axes[1, col]
        a1.plot(t, ld(t, szz_T), '-', color=C_TRENCH, lw=2.0,
                label=f'trench column $\\bar\\sigma_{{zz}}(x_T)$  '
                      f'(roughness {rough(ld(t, szz_T)):.2f})')
        a1.plot(t, ld(t, szz_I), '-', color=C_XI, lw=2.0,
                label=f'first isostatic column $\\bar\\sigma_{{zz}}(x_I)$  '
                      f'(roughness {rough(ld(t, szz_I)):.2f})')
        a1.axhline(0, color='k', lw=0.8)
        a1.set_ylabel('column integral,\ndetrended [TN/m]', fontsize=9.5)
        a1.legend(frameon=False, fontsize=8.5, loc='upper left')

        # --- row 3: stabilised-reference PROXY (see docstring) ----------
        # The x_I column PROFILE is median-filtered through time at every
        # depth, then the trench pull is re-formed. This does not re-pick
        # x_I -- it removes what an unstable pick injects.
        p_I_stab = median_filter(d[f'{key}_szz_I'], size=(3, 1), mode='nearest')
        TP_stab = I(d[f'{key}_szz_T']) - I(p_I_stab)
        a2 = axes[2, col]
        a2.plot(t, TP, '-', color=C_RAW, lw=2.6, alpha=0.7,
                label=f'raw  (roughness {rough(ld(t, TP)):.2f}, '
                      f's.d. {ld(t, TP).std():.3f})')
        a2.plot(t, TP_stab, '-', color=C_STAB, lw=2.0,
                label=f'stabilised $x_I$ reference (proxy)  '
                      f'(roughness {rough(ld(t, TP_stab)):.2f}, '
                      f's.d. {ld(t, TP_stab).std():.3f})')
        a2.set_ylabel('trench pull [TN/m]', fontsize=10)
        a2.legend(frameon=False, fontsize=8.5, loc='upper left')

        # --- row 4: Dan's question about V ------------------------------
        V, ND = r['v_T'] / 1e12, r['nd_T'] / 1e12
        a3 = axes[3, col]
        a3.plot(t, ld(t, ND), '-', color='k', lw=2.0,
                label=f'$N_D(x_T)$  (roughness {rough(ld(t, ND)):.2f}, '
                      f's.d. {ld(t, ND).std():.3f})')
        a3.plot(t, ld(t, V), '-', color=C_V, lw=2.0,
                label=f'$V(x_T)$  (roughness {rough(ld(t, V)):.2f}, '
                      f's.d. {ld(t, V).std():.3f})')
        a3.plot(t, ld(t, TP), '-', color=C_RAW, lw=1.6, alpha=0.8,
                label=f'trench pull  (roughness {rough(ld(t, TP)):.2f}, '
                      f's.d. {ld(t, TP).std():.3f})')
        a3.axhline(0, color='k', lw=0.8)
        a3.set_ylabel('detrended [TN/m]\n(ABSOLUTE — not normalised)', fontsize=9.5)
        a3.set_xlabel('Model time [Myr]', fontsize=11)
        a3.legend(frameon=False, fontsize=8.5, loc='upper left')

        for ax in axes[:, col]:
            ax.grid(alpha=0.25, color=C_RULE, lw=0.6)

        print(f'{key}: TP raw roughness {rough(ld(t, TP)):.2f} '
              f'(HF {hf_share(t, ld(t, TP)):.0f} %), stabilised '
              f'{rough(ld(t, TP_stab)):.2f} (HF {hf_share(t, ld(t, TP_stab)):.0f} %); '
              f's.d. {ld(t, TP).std():.3f} -> {ld(t, TP_stab).std():.3f} '
              f'({100*ld(t, TP_stab).std()/ld(t, TP).std():.0f} % retained); '
              f'szz_T roughness {rough(ld(t, szz_T)):.2f}, szz_I '
              f'{rough(ld(t, szz_I)):.2f}; V {rough(ld(t, V)):.2f} '
              f'(s.d. {ld(t, V).std():.3f}), N_D(x_T) {rough(ld(t, ND)):.2f} '
              f'(s.d. {ld(t, ND).std():.3f})')
        rows += [(key, 'roughness_trench_pull_raw', f'{rough(ld(t, TP)):.3f}'),
                 (key, 'roughness_trench_pull_stabilised', f'{rough(ld(t, TP_stab)):.3f}'),
                 (key, 'hf_share_trench_pull_raw_percent', f'{hf_share(t, ld(t, TP)):.1f}'),
                 (key, 'hf_share_trench_pull_stabilised_percent',
                  f'{hf_share(t, ld(t, TP_stab)):.1f}'),
                 (key, 'sd_trench_pull_raw_TNm', f'{ld(t, TP).std():.4f}'),
                 (key, 'sd_trench_pull_stabilised_TNm', f'{ld(t, TP_stab).std():.4f}'),
                 (key, 'amplitude_retained_percent',
                  f'{100*ld(t, TP_stab).std()/ld(t, TP).std():.1f}'),
                 (key, 'roughness_szz_trench_column', f'{rough(ld(t, szz_T)):.3f}'),
                 (key, 'roughness_szz_first_isostatic_column', f'{rough(ld(t, szz_I)):.3f}'),
                 (key, 'roughness_V_trench', f'{rough(ld(t, V)):.3f}'),
                 (key, 'roughness_nd_trench', f'{rough(ld(t, ND)):.3f}'),
                 (key, 'sd_V_trench_TNm', f'{ld(t, V).std():.4f}'),
                 (key, 'sd_nd_trench_TNm', f'{ld(t, ND).std():.4f}')]

    # STD and WAL share a y scale row by row (Dan, 2026-09-23) so the two
    # models can be compared by eye rather than by reading the ticks. The
    # detrended rows (2 and 4) are additionally forced symmetric about
    # zero -- they are anomalies, and an asymmetric anomaly axis
    # misrepresents which way the excursions go.
    for row in range(4):
        pair = axes[row, :]
        lo = min(ax.get_ylim()[0] for ax in pair)
        hi = max(ax.get_ylim()[1] for ax in pair)
        if row in (1, 3):
            hi = max(abs(lo), abs(hi)); lo = -hi
        for ax in pair:
            ax.set_ylim(lo, hi)

    fig.suptitle('Diagnostic — the trench pull\'s jaggedness is the $x_I$ reference, '
                 'not the trench column', fontsize=12)
    fig.tight_layout()
    out = os.path.join(ROOT, 'figures', 'diag_trench_pull_smoothing.png')
    fig.savefig(out, bbox_inches='tight', dpi=200)
    print('written:', out)
    tab = write_table('diag_trench_pull_smoothing', rows[0], rows[1:],
                      script='diag_trench_pull_smoothing.py',
                      figure='diag_trench_pull_smoothing.png',
                      models=('STD', 'WAL'),
                      meta={'roughness': 'sd(step difference)/sd of the detrended series',
                            'hf_cut_Myr': 12.0,
                            'stabilised': 'PROXY — x_I column profile median-filtered '
                                          'through time (size 3); NOT a re-pick',
                            'note': 'diagnostic, not a manuscript figure'})
    print('written:', tab)


if __name__ == '__main__':
    main()
