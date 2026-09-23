"""diag_trench_pull_spectrum — does the trench pull carry power at the slab's frequency?

DIAGNOSTIC (diag_, not a manuscript figure). Writes
figures/diag_trench_pull_spectrum.png and tables/diag_trench_pull_spectrum.csv.

Dan's question, 2026-09-23, stated plainly: *is there significant power in
the trench pull variation at the same frequency as the slab buckling?*

This is the question the whole x_I thread was in the way of. The raw
trench pull has 51 % (STD) of its variance at periods below 12 Myr and no
clean spectral peak, so a periodogram of it answers nothing. The
stabilised series (diag_trench_pull_smoothing row 3 — the x_I column
profile median-filtered through time, a proxy for a stable pick) is what
gets tested here, with the raw series shown beside it so the dilution is
visible rather than assumed.

WHAT "THE SLAB'S FREQUENCY" MEANS HERE, AND ITS LIMIT. Two slab series
are used: the upper-mantle descent rate and the dip. ⚠ THE DIP IS NOT A
BUCKLING MEASURE. It is a linear fit to the cold-anomaly centroid over
200-400 km, so it measures mean inclination; folding would be a
CURVATURE, which is not yet computed (DYNAMICS_FINDINGS §4.6). What is
tested is therefore "the frequency at which the slab's descent and
inclination vary", which is the observable the buckling interpretation
rests on, not buckling itself.

SAMPLING LIMIT, STATED UP FRONT. 37 snapshots at 2 Myr: the record is
72 Myr and the Nyquist period is 4 Myr, so the Fourier bins sit at 72,
36, 24, 18, 14.4, 12 Myr ... The ~25 Myr signal is bin 3. There is no
frequency resolution to speak of and no amount of analysis creates any:
a band is the finest statement available, and the band used is
18-36 Myr (bins 2-4).

THE TEST. Two separate questions, kept separate:

  1  EXCESS POWER. What share of each series' variance sits in the
     18-36 Myr band? White noise would put 3/18 = 17 % there, so that is
     the null line drawn on the bars.
  2  PHASE ALIGNMENT. Band-limited correlation between the trench pull and
     each slab series, with a p-value from 5,000 PHASE-RANDOMISED
     SURROGATES of the trench pull. Surrogates preserve its amplitude
     spectrum exactly and destroy only its phase relationship with the
     slab, so this asks whether the alignment is better than chance GIVEN
     both spectra. AR(1) is not used and must not be (§5.4).

  1 without 2 is not enough: two series can both peak at 25 Myr and be in
  quadrature, which would mean they are not coupled at that frequency.

  1  periodograms, variance-normalised, with the band shaded.
  2  the band-passed series overlaid.
  3  band-power shares against the white-noise null, and the band
     correlations with their surrogate p-values.

THE ANSWER FOR STD IS NO, AND IT IS EMPHATIC.

  share of variance in the 18-36 Myr band (white-noise null 11 %):

                              STD      WAL
    slab descent rate        66 %     33 %
    slab dip                 69 %     46 %
    Delta N_D                68 %     51 %
    trench pull, stabilised  20 %     40 %
    trench pull, raw         15 %     31 %

  and the single-bin form, which is sharper still. In STD the slab's
  descent rate peaks in the 25 Myr bin with 42 % of its variance. In that
  SAME bin: slab dip 42 %, Delta N_D 53 % -- and the trench pull 4.4 %.
  The trench pull is not merely weak at the slab's frequency, it sits at a
  local MINIMUM of its own spectrum there. Its power is at the record
  length (24 %) and scattered through 12-14 Myr.

  WAL does not rescue the idea. Its slab descent peaks at 15 Myr, its
  trench pull at 36 Myr; both have band power but at different periods
  inside the band, and the peak-bin shares are 22 % against 12 %.

  No band correlation reaches significance in either run (best: STD
  Delta N_D against slab descent, r = -0.90, p = 0.065). But the p-values
  are the WEAKEST part of this figure, not the strongest -- with three
  Fourier bins in the band and n = 37 the surrogate test has very little
  power, which is exactly why r = -0.90 fails to clear 0.05. Read the
  variance shares, which need no test: 68 % against 20 % is not a
  marginal difference.

WHAT IT MEANS. The slab's oscillation reaches the trailing plate through
Delta N_D and essentially not at all through the trench pull. That is the
same conclusion as DYNAMICS_FINDINGS §1.7 -- which found velocity changes
paid for by Delta N_D relaxing, with the driving term unmoved -- arrived
at independently and in the frequency domain. The two now agree.

⚠ It also settles the reading of fig_slab_correlations that prompted this:
the trench pull's apparent shared periodicity with the slab was the eye
matching envelopes on a normalised axis. It does not survive a spectrum.
And the x_I stabilisation does not change the verdict -- 20 % against
15 % -- so this is not a measurement artefact either.

DRAFT CAPTION (if needed). Spectral test of the trench pull against the
slab, STD (left) and WAL (right). Top: periodograms of the slab descent
rate, slab dip, stabilised trench pull and raw trench pull, each
normalised by its own variance, with the 18-36 Myr band shaded. Middle:
the same series band-passed to that band. Bottom: share of variance in
the band, against the white-noise expectation, and band-limited
correlations with p-values from phase-randomised surrogates.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.join(_HERE, 'kinematics'))
import column_profiles_cache as cpc
import slab_geometry_cache as sgc
from fig_budget_time import terms
from tables_io import write_table

ROOT = os.path.dirname(_HERE)
BAND = (18.0, 36.0)          # periods [Myr]; bins 2-4 of a 72 Myr record
N_SURR = 5000
SEED = 20260923
C_RULE = '#BFC3D1'
C = {'slab descent rate': '#0072B2', 'slab dip': '#56B4E9',
     'trench pull (stabilised)': '#1B9E77', 'trench pull (raw)': '0.62',
     '$\\Delta N_D$': 'k'}


def uniform(t, y):
    """Detrend and resample onto a uniform grid (dt varies by ~2 %)."""
    y = y - np.polyval(np.polyfit(t, y, 1), t)
    tu = np.linspace(t[0], t[-1], len(t))
    return tu, np.interp(tu, t, y)


def spectrum(tu, y):
    P = np.abs(np.fft.rfft(y - y.mean())) ** 2
    f = np.fft.rfftfreq(len(y), tu[1] - tu[0])
    return f[1:], P[1:] / P[1:].sum()          # drop DC, normalise to unit total


def band_share(tu, y):
    f, P = spectrum(tu, y)
    per = 1.0 / f
    return float(P[(per >= BAND[0]) & (per <= BAND[1])].sum())


def bandpass(tu, y):
    F = np.fft.rfft(y - y.mean())
    f = np.fft.rfftfreq(len(y), tu[1] - tu[0])
    per = np.divide(1.0, f, out=np.full_like(f, np.inf), where=f > 0)
    F[~((per >= BAND[0]) & (per <= BAND[1]))] = 0
    return np.fft.irfft(F, n=len(y))


def surrogate(y, rng):
    """Phase-randomised surrogate: identical amplitude spectrum, random phases."""
    F = np.fft.rfft(y - y.mean())
    ph = rng.uniform(0, 2 * np.pi, len(F))
    ph[0] = 0.0
    if len(y) % 2 == 0:
        ph[-1] = 0.0
    return np.fft.irfft(np.abs(F) * np.exp(1j * ph), n=len(y))


def main():
    if not hasattr(np, 'trapezoid'):
        np.trapezoid = np.trapz
    d, s = cpc.load(), sgc.load()
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(3, 2, figsize=(13.2, 11.4),
                             gridspec_kw={'height_ratios': [1.25, 1.0, 1.0]})
    rows = [('model', 'quantity', 'value')]

    for col, key in enumerate(('STD', 'WAL')):
        c = cpc.derive(d, key); q = terms(d, key)
        z, zc, t = c['z'], c['zc'], c['t']
        ts = s[f'{key}_t']; n = min(len(t), len(ts)); t = t[:n]
        I = lambda a: np.trapezoid(a[:n, zc], z[zc], axis=1) / 1e12
        TP_raw = -I(c['p_T'])
        TP_stab = I(d[f'{key}_szz_T']) - I(median_filter(d[f'{key}_szz_I'],
                                                         size=(3, 1), mode='nearest'))
        series = {
            'slab descent rate': s[f'{key}_vz_upper'][:n],
            'slab dip': s[f'{key}_dip'][:n],
            'trench pull (stabilised)': TP_stab,
            'trench pull (raw)': TP_raw,
            '$\\Delta N_D$': q['d_nd'][:n] / 1e12,
        }
        U = {nm: uniform(t, y) for nm, y in series.items()}
        tu = U['slab descent rate'][0]
        nbin = len(np.fft.rfftfreq(len(tu))) - 1
        per_all = 1.0 / np.fft.rfftfreq(len(tu), tu[1] - tu[0])[1:]
        null = ((per_all >= BAND[0]) & (per_all <= BAND[1])).sum() / nbin

        # --- row 1: periodograms ----------------------------------------
        a0 = axes[0, col]
        a0.axvspan(BAND[0], BAND[1], color='#FFE9B0', alpha=0.55, zorder=0,
                   label=f'{BAND[0]:.0f}–{BAND[1]:.0f} Myr band')
        for nm, (tux, y) in U.items():
            f, P = spectrum(tux, y)
            a0.plot(1 / f, P, 'o-', color=C[nm], ms=3.5,
                    lw=2.4 if 'slab descent' in nm else 1.6,
                    alpha=0.55 if 'raw' in nm else 1.0, label=nm)
        a0.set_xscale('log'); a0.set_xlim(72, 4)
        a0.set_xticks([72, 36, 24, 18, 12, 8, 6, 4])
        a0.set_xticklabels(['72', '36', '24', '18', '12', '8', '6', '4'])
        a0.set_xlabel('Period [Myr]   (Fourier bins; no resolution between them)',
                      fontsize=9.5)
        a0.set_ylabel('share of variance', fontsize=10)
        a0.set_title(f'{key}', fontsize=11)
        a0.legend(frameon=False, fontsize=8, loc='upper right')

        # --- row 2: band-passed ------------------------------------------
        a1 = axes[1, col]
        for nm in ('slab descent rate', 'trench pull (stabilised)',
                   'trench pull (raw)', '$\\Delta N_D$'):
            tux, y = U[nm]
            bp = bandpass(tux, y)
            a1.plot(tux, bp / y.std(), '-', color=C[nm],
                    lw=2.4 if 'slab' in nm else 1.8,
                    alpha=0.5 if 'raw' in nm else 1.0, label=nm)
        a1.axhline(0, color='k', lw=0.8)
        a1.set_ylabel(f'band-passed {BAND[0]:.0f}–{BAND[1]:.0f} Myr\n'
                      '(÷ full s.d.)', fontsize=9)
        a1.set_xlabel('Model time [Myr]', fontsize=10.5)
        a1.legend(frameon=False, fontsize=8, loc='upper left', ncol=2)

        # --- row 3: band share + surrogate-tested band correlation --------
        a2 = axes[2, col]
        names = list(series)
        shares = [band_share(*U[nm]) for nm in names]
        a2.barh(names, shares, color=[C[nm] for nm in names], height=0.6)
        a2.axvline(null, color='r', ls='--', lw=1.4,
                   label=f'white-noise null ({100*null:.0f} %)')
        a2.set_xlabel(f'share of variance in the {BAND[0]:.0f}–{BAND[1]:.0f} Myr band',
                      fontsize=9.5)
        a2.tick_params(axis='y', labelsize=8)
        a2.legend(frameon=False, fontsize=8, loc='lower right')
        for i, v in enumerate(shares):
            a2.text(v + 0.008, i, f'{100*v:.0f} %', va='center', fontsize=8)

        print(f'\n=== {key}   band {BAND[0]:.0f}-{BAND[1]:.0f} Myr, '
              f'white-noise null {100*null:.0f} %')
        for nm, sh in zip(names, shares):
            print(f'   band share  {nm:26s} {100*sh:5.1f} %'
                  f'{"   <== below null" if sh < null else ""}')
            rows.append((key, f'band_share_{nm.strip("$").replace(" ", "_").replace("\\", "")}',
                         f'{sh:.4f}'))
        rows.append((key, 'white_noise_null_share', f'{null:.4f}'))

        # THE SHARPEST FORM OF THE ANSWER: the single Fourier bin where the
        # slab's descent rate peaks, and what every other series has there.
        f_s, P_s = spectrum(*U['slab descent rate'])
        kpk = int(np.argmax(P_s)); per_pk = 1.0 / f_s[kpk]
        print(f'   slab descent peaks in the {per_pk:.0f} Myr bin '
              f'({100*P_s[kpk]:.0f} % of its variance). In that same bin:')
        for nm in names:
            P = spectrum(*U[nm])[1]
            print(f'      {nm:26s} {100*P[kpk]:5.1f} %')
            rows.append((key, f'peak_bin_share_{nm.strip("$").replace(" ", "_").replace("\\", "")}',
                         f'{P[kpk]:.4f}'))
        rows.append((key, 'slab_peak_period_Myr', f'{per_pk:.1f}'))

        # phase-alignment test
        for tgt in ('slab descent rate', 'slab dip'):
            bt = bandpass(*U[tgt])
            for nm in ('trench pull (stabilised)', 'trench pull (raw)',
                       '$\\Delta N_D$'):
                tux, y = U[nm]
                r = float(np.corrcoef(bandpass(tux, y), bt)[0, 1])
                null_r = np.array([np.corrcoef(bandpass(tux, surrogate(y, rng)),
                                               bt)[0, 1] for _ in range(N_SURR)])
                p = float((np.abs(null_r) >= abs(r)).mean())
                star = '  *' if p < 0.05 else ''
                print(f'   band r({tgt:18s}, {nm:26s}) = {r:+.2f}   '
                      f'p = {p:.3f}{star}')
                sl = lambda x: x.strip('$').replace(' ', '_').replace('\\', '') \
                                .replace('(', '').replace(')', '')
                rows += [(key, f'band_corr_{sl(tgt)}_{sl(nm)}', f'{r:.3f}'),
                         (key, f'band_corr_p_{sl(tgt)}_{sl(nm)}', f'{p:.4f}')]

        for ax in axes[:, col]:
            ax.grid(alpha=0.25, color=C_RULE, lw=0.6)

    for row in (1,):
        lo = min(ax.get_ylim()[0] for ax in axes[row, :])
        hi = max(ax.get_ylim()[1] for ax in axes[row, :])
        hi = max(abs(lo), abs(hi))
        for ax in axes[row, :]:
            ax.set_ylim(-hi, hi)

    fig.suptitle('Does the trench pull carry power at the slab\'s frequency? '
                 f'({N_SURR:,} phase-randomised surrogates)', fontsize=11.5)
    fig.tight_layout()
    out = os.path.join(ROOT, 'figures', 'diag_trench_pull_spectrum.png')
    fig.savefig(out, bbox_inches='tight', dpi=200)
    print('\nwritten:', out)
    tab = write_table('diag_trench_pull_spectrum', rows[0], rows[1:],
                      script='diag_trench_pull_spectrum.py',
                      figure='diag_trench_pull_spectrum.png',
                      models=('STD', 'WAL'),
                      meta={'band_Myr': list(BAND), 'n_surrogates': N_SURR,
                            'seed': SEED,
                            'sampling': '37 snapshots at 2 Myr; bins 72/36/24/18/...',
                            'surrogates': 'phase-randomised, amplitude spectrum preserved',
                            'caveat': 'dip is mean inclination, NOT curvature — '
                                      'not a buckling measure (§4.6)',
                            'note': 'diagnostic, not a manuscript figure'})
    print('written:', tab)


if __name__ == '__main__':
    main()
