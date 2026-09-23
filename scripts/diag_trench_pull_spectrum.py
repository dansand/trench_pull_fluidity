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

WHAT "THE SLAB'S FREQUENCY" MEANS HERE. Two slab series are used: the
upper-mantle descent rate and the mean dip. Dan's ruling 2026-09-23: that
pair is sufficient -- a coherent pattern in mean dip and mid-upper-mantle
sinking rate is evidence of buckling, no curvature diagnostic is required,
and the buckling interpretation of this model is already established in
the earlier paper. It is inherited here, not re-derived, so "the slab's
buckling frequency" is written plainly. (An earlier draft of this
docstring hedged on that; the hedge is withdrawn.)

SAMPLING LIMIT, AND WHY THE BAND WAS A BAD SUMMARY. 37 snapshots at
2 Myr: the record is 72 Myr and the Nyquist period is 4 Myr, so the
Fourier bins sit at 74, 37, 24.7, 18.5, 14.8, 12.3, 10.6 Myr ... There is
no frequency resolution between them and no analysis creates any.

⚠ The first version of this figure summarised with an 18-36 Myr band.
That band contains exactly TWO bins -- 24.7 and 18.5 -- because 37.0 falls
just outside it. Widening to 15-35 Myr changes nothing at all (same two
bins); 10-40 Myr admits six but also raises the white-noise null from 11 %
to 33 % and mixes in the 10-14 Myr content, which is a different
phenomenon. A two-bin band-pass is two sinusoids, and two sinusoids look
quasi-periodic whatever the input -- Dan raised exactly this on
2026-09-23 and he is right. THE PER-BIN TABLE IS THE HONEST STATEMENT;
the band is retained only as a one-number summary with its bin count
printed.

THE TEST. Three separate questions, kept separate:

  1  EXCESS POWER. What share of each series' variance sits in the
     18-36 Myr band? The null is white noise, which spreads variance
     evenly over bins, so it is (bins in band)/(bins total) -- computed
     from the actual grid rather than assumed, and drawn on the bars. It
     comes out at 11 % for both runs.
  2  PHASE ALIGNMENT. Band-limited correlation between the trench pull and
     each slab series, with a p-value from 5,000 PHASE-RANDOMISED
     SURROGATES of the trench pull. Surrogates preserve its amplitude
     spectrum exactly and destroy only its phase relationship with the
     slab, so this asks whether the alignment is better than chance GIVEN
     both spectra. AR(1) is not used and must not be (§5.4).

  3  IS THE BAND-PASSED PICTURE AN ILLUSION? The observed two-bin
     reconstruction is drawn against reconstructions of phase-randomised
     surrogates of the SAME series. If the surrogates look equally
     oscillatory and equally "phase-shifted", the appearance carries no
     information -- which is the control the first version lacked.

  1 without 2 is not enough: two series can both peak at 25 Myr and be in
  quadrature, which would mean they are not coupled at that frequency.

  1  per-bin periodograms as BARS (there is nothing between the bins, so
     a connecting line overstates the resolution), slab peak bin marked.
  2  SINGLE-BIN reconstruction at the slab's peak bin. One sinusoid per
     series, so amplitude and phase are read directly and no illusion is
     possible.
  3  the two-bin band-pass against surrogate reconstructions -- the
     illusion control.

THE ANSWER, REVISED 2026-09-23 AFTER DAN CHALLENGED THE BAND-PASS.
The first version said "no, emphatically". The per-bin view says something
more precise and more interesting: SMALL BUT IN PHASE.

  STD. The trench pull's own peak is at 74 Myr -- the record length, i.e.
  residual curvature after linear detrending, not an oscillation. Its
  spectrum near the slab's 24.7 Myr peak reads

      74 Myr  25 %  <- its peak
      37      11
      24.7     4.4  <- the slab's peak; a LOCAL MINIMUM of the trench pull
      18.5    15
      14.8    12
      10.6    13

  against slab descent 42 %, slab dip 42 %, Delta N_D 53 % and F_B 31 % in
  that same 24.7 Myr bin. But share of variance is not force. In absolute
  terms the 24.7 Myr component is

      Delta N_D     amplitude 0.296 TN/m, lag -10.4 Myr -- near ANTIPHASE
                    (the r = -0.90 of the band correlation)
      F_B           amplitude 0.243 TN/m, lag  +1.7 Myr -- near IN PHASE
      trench pull   amplitude 0.037 TN/m, lag  -1.6 Myr -- in phase, tiny

  ⚠ F_B WAS ADDED AT DAN'S REQUEST 2026-09-23 AND IT COMPLETES THE
  PICTURE. It is also the ONLY term whose band correlation with the slab
  clears significance: r = +0.93, p = 0.049, against a surrogate median
  |r| of 0.42.

  THE BUDGET AT THE BUCKLING FREQUENCY (STD, amplitudes in TN/m):

      drive 0.038  =  Delta N_D 0.296  +  F_B 0.243

  which balances only because Delta N_D and F_B are 178 DEGREES APART --
  their vector sum is 0.054, an order of magnitude below either. The
  buckling cycle is almost entirely an EXCHANGE BETWEEN THE TWO
  RESISTANCE TERMS at nearly constant drive. The trench pull's 0.037 is
  the same size as the whole drive fluctuation, and both are ~13 % of the
  terms being exchanged.

  So the answer is not "no response" but: the trench pull responds in
  phase, at an eighth of Delta N_D's amplitude, while the cycle itself is
  carried by Delta N_D trading against F_B.

  WAL is a different animal and should not be averaged with STD. Its slab
  descent peaks at 14.8 Myr while its dip peaks at 24.6; its trench pull
  peaks at 37 Myr. At the descent peak the trench pull's amplitude
  (0.131 TN/m) is comparable to Delta N_D's (0.156), so WAL does NOT
  reproduce STD's eightfold separation. The exchange structure does
  survive, more weakly: Delta N_D 0.156 against F_B 0.181, 150 deg apart,
  vector sum 0.090 against a drive of 0.083; F_B again the best-correlated
  term (r = +0.77, p = 0.095).

⚠ DAN'S PHASE-SHIFT READING, AND WHY THE BAND-PASS MISLED. In the
two-bin band-pass the trench pull looked like the slab's frequency shifted
in phase. It is not: at 24.7 Myr the lag is only -1.6 Myr. What produced
the appearance is that the band-passed trench pull is dominated by its
18.5 Myr component (15 %) rather than its 24.7 Myr one (4.4 %), and an
18.5 Myr sinusoid drifts steadily against a 24.7 Myr one -- across three
cycles that drift reads to the eye as a fixed lag.

⚠ AND THE ILLUSION IS REAL, QUANTIFIED. Pushed through this two-bin
filter, PHASE-RANDOMISED SURROGATES of the trench pull correlate with the
band-passed slab at a median |r| of 0.33 (STD) / 0.36 (WAL). The observed
value is +0.43. Row 3 draws eight surrogates beside the observed series:
they oscillate just as convincingly. Any narrow band-pass manufactures
apparent coherence, and this one is about as narrow as a filter can be.

ON WIDENING THE BAND (Dan's suggestion). 15-35 Myr selects the IDENTICAL
two bins and changes nothing whatsoever. 10-40 Myr admits six bins, but
the white-noise null rises from 11 % to 33 % and the band then includes
the 10-14 Myr content, which is a separate phenomenon -- STD's trench
pull jumps to 57 % on that band purely by absorbing power that has
nothing to do with the slab. Neither helps. The per-bin table above is
the statement that survives; a band is only ever a lossy summary of it.

WHAT IT MEANS. The buckling cycle reaches the trailing plate as an
exchange between the in-plane resultant and basal drag, at a drive that
is very nearly constant: when the slab descends faster the plate speeds
up, basal drag rises, and the in-plane compression relaxes by almost
exactly the same amount. The trench pull participates only weakly, in
phase, at an eighth of that amplitude.

This is DYNAMICS_FINDINGS §1.7 (76 % of a slab pulse absorbed by
Delta N_D relaxing, 15 % by trench pull) recovered independently in the
frequency domain, and the two agree numerically: the 8:1 amplitude ratio
here against the 76:15 split found there by regression.

DRAFT CAPTION (if needed). Spectral test of the trench pull against the
slab, STD (left) and WAL (right). Top: periodograms of the slab descent
rate, slab dip, stabilised trench pull and raw trench pull, each
normalised by its own variance and drawn as bars, since the record
supports no resolution between bins; the slab's peak bin is shaded.
Middle: each series reconstructed from that single bin alone, scaled by
its own standard deviation, so amplitude and phase are directly
comparable. Bottom: the two-bin band-pass of the trench pull against
eight phase-randomised surrogates of itself, which oscillate equally
convincingly -- the control showing that a narrow band-pass manufactures
apparent coherence.
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
     '$\\Delta N_D$': 'k', '$F_B$': 'red'}


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


def onebin(tu, y, k):
    """Reconstruct y from the SINGLE Fourier bin k (1-based on the DC-dropped
    grid). One sinusoid, so amplitude and phase are read directly and the
    two-sinusoid beating that makes a narrow band-pass look oscillatory
    cannot occur."""
    F = np.fft.rfft(y - y.mean())
    G = np.zeros_like(F); G[k + 1] = F[k + 1]
    return np.fft.irfft(G, n=len(y))


def main():
    if not hasattr(np, 'trapezoid'):
        np.trapezoid = np.trapz
    d, s = cpc.load(), sgc.load()
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(3, 2, figsize=(13.6, 12.2),
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
            '$\\Delta N_D$': q['d_nd'][:n] / 1e12,
            '$F_B$': q['f_b'][:n] / 1e12,
            'trench pull (stabilised)': TP_stab,
            'trench pull (raw)': TP_raw,
        }
        U = {nm: uniform(t, y) for nm, y in series.items()}
        tu = U['slab descent rate'][0]
        per_all = 1.0 / np.fft.rfftfreq(len(tu), tu[1] - tu[0])[1:]
        nbin = len(per_all)
        inband = (per_all >= BAND[0]) & (per_all <= BAND[1])
        null = inband.sum() / nbin
        P = {nm: spectrum(*U[nm])[1] for nm in series}
        kpk = int(np.argmax(P['slab descent rate']))

        # --- row 1: per-bin bars ----------------------------------------
        NB = 9
        a0 = axes[0, col]
        w = 0.8 / len(series)
        xs = np.arange(NB)
        for i, nm in enumerate(series):
            a0.bar(xs + (i - (len(series) - 1) / 2) * w, 100 * P[nm][:NB], w,
                   color=C[nm], label=nm,
                   alpha=0.55 if 'raw' in nm else 1.0)
        a0.axvspan(kpk - 0.5, kpk + 0.5, color='#FFE9B0', alpha=0.5, zorder=0)
        a0.text(kpk, a0.get_ylim()[1] * 0.97,
                f"slab's peak\n{per_all[kpk]:.0f} Myr", ha='center', va='top',
                fontsize=8.5, color='0.35')
        a0.set_xticks(xs)
        a0.set_xticklabels([f'{p:.0f}' for p in per_all[:NB]])
        a0.set_xlabel('Fourier bin, labelled by period [Myr] — nothing exists '
                      'between the bars', fontsize=9)
        a0.set_ylabel('share of variance [%]', fontsize=10)
        a0.set_title(f'{key}', fontsize=11)
        a0.legend(frameon=False, fontsize=8, loc='upper right')

        # --- row 2: SINGLE-BIN reconstruction at the slab's peak --------
        a1 = axes[1, col]
        ref = onebin(tu, U['slab descent rate'][1], kpk)
        for nm in series:
            tux, y = U[nm]
            ob = onebin(tux, y, kpk)
            a1.plot(tux, ob / y.std(), '-', color=C[nm],
                    lw=2.6 if 'slab descent' in nm else 1.9,
                    alpha=0.5 if 'raw' in nm else 1.0,
                    label=f'{nm}  ({100*P[nm][kpk]:.0f} %)')
        a1.axhline(0, color='k', lw=0.8)
        a1.set_ylabel(f'{per_all[kpk]:.0f} Myr component\n(÷ full s.d. of each)',
                      fontsize=9)
        a1.set_xlabel('Model time [Myr]', fontsize=10.5)
        a1.legend(frameon=False, fontsize=7.5, loc='upper left', ncol=2)
        a1.set_title('one sinusoid each — amplitude is directly comparable',
                     fontsize=9, color='0.3')

        # --- row 3: the illusion control --------------------------------
        a2 = axes[2, col]
        tux, y = U['trench pull (stabilised)']
        for j in range(8):
            a2.plot(tux, bandpass(tux, surrogate(y, rng)) / y.std(), '-',
                    color='#B0B0B0', lw=1.0, alpha=0.75,
                    label='phase-randomised surrogates (8)' if j == 0 else None)
        a2.plot(tux, bandpass(*U['slab descent rate'])
                / U['slab descent rate'][1].std(), '-', color=C['slab descent rate'],
                lw=2.6, label='slab descent rate')
        a2.plot(tux, bandpass(tux, y) / y.std(), '-',
                color=C['trench pull (stabilised)'], lw=2.4,
                label='trench pull (stabilised), observed')
        a2.axhline(0, color='k', lw=0.8)
        a2.set_ylabel(f'{BAND[0]:.0f}–{BAND[1]:.0f} Myr band-pass\n(÷ full s.d.)',
                      fontsize=9)
        a2.set_xlabel('Model time [Myr]', fontsize=10.5)
        a2.legend(frameon=False, fontsize=8, loc='upper left', ncol=2)
        a2.set_title(f'illusion control — the band holds only {inband.sum()} bins; '
                     'surrogates oscillate just as convincingly',
                     fontsize=9, color='0.3')

        # ---- numbers ----------------------------------------------------
        print(f'\n=== {key}   bins: '
              + ', '.join(f'{p:.1f}' for p in per_all[:7])
              + f' ...   band {BAND[0]:.0f}-{BAND[1]:.0f} Myr holds '
                f'{inband.sum()} bins (null {100*null:.0f} %)')
        # Amplitude of the slab-bin component in PHYSICAL units, and its
        # phase relative to the slab. For a single Fourier component that
        # carries a fraction f of a series' variance, the amplitude is
        # sqrt(2 f) * sd -- so a small share of a large series can still be
        # a large force, and this is the check on that.
        Fslab = np.fft.rfft(U['slab descent rate'][1]
                            - U['slab descent rate'][1].mean())[kpk + 1]
        for nm in series:
            kp = int(np.argmax(P[nm]))
            sd = U[nm][1].std()
            amp = np.sqrt(2 * P[nm][kpk]) * sd
            Fk = np.fft.rfft(U[nm][1] - U[nm][1].mean())[kpk + 1]
            dphi = np.angle(Fk / Fslab)
            lag = dphi / (2 * np.pi) * per_all[kpk]
            print(f'   {nm:26s} peak {per_all[kp]:5.1f} Myr ({100*P[nm][kp]:4.1f} %)'
                  f'   at the slab bin {100*P[nm][kpk]:5.1f} %'
                  f'   band {100*P[nm][inband].sum():5.1f} %'
                  f'   amp {amp:7.3f}   lag {lag:+5.1f} Myr')
            sl = nm.strip('$').replace(' ', '_').replace('\\', '') \
                   .replace('(', '').replace(')', '')
            rows += [(key, f'peak_period_{sl}_Myr', f'{per_all[kp]:.1f}'),
                     (key, f'peak_share_{sl}', f'{P[nm][kp]:.4f}'),
                     (key, f'slab_bin_share_{sl}', f'{P[nm][kpk]:.4f}'),
                     (key, f'band_share_{sl}', f'{P[nm][inband].sum():.4f}'),
                     (key, f'slab_bin_amplitude_{sl}', f'{amp:.4f}'),
                     (key, f'slab_bin_lag_vs_slab_{sl}_Myr', f'{lag:.2f}')]
        rows += [(key, 'slab_peak_period_Myr', f'{per_all[kpk]:.1f}'),
                 (key, 'band_bin_count', str(int(inband.sum()))),
                 (key, 'white_noise_null_share', f'{null:.4f}')]

        # THE BUDGET AT THE BUCKLING FREQUENCY. The Fourier transform is
        # linear, so if the balance dN_D - dGPE* + F_B = 0 holds in time it
        # holds bin by bin. This is therefore arithmetic, not a new test --
        # but it is the arithmetic that says which term absorbs the cycle.
        Fk = lambda y: np.fft.rfft(y - y.mean())[kpk + 1]
        gpe = Fk(uniform(t, q['d_gpe'][:n] / 1e12)[1])
        nd, fb = Fk(U['$\\Delta N_D$'][1]), Fk(U['$F_B$'][1])
        # Peak amplitude of a real Fourier component is 2|F_k|/N. (An
        # earlier version used an extra sqrt(2) here and inflated every
        # amplitude by 41 %; it is cross-checked below against the
        # independent route A = sqrt(2 f) * sd.)
        sc = 2.0 / len(tu)
        print(f'   BUDGET at {per_all[kpk]:.0f} Myr [TN/m amplitude]: '
              f'drive {abs(gpe)*sc:.3f}  =  dN_D {abs(nd)*sc:.3f} + F_B '
              f'{abs(fb)*sc:.3f}   (vector sum {abs(nd + fb)*sc:.3f}; '
              f'dN_D and F_B are {np.degrees(np.angle(nd / fb)):+.0f} deg apart)')
        rows += [(key, 'bin_amplitude_drive_TNm', f'{abs(gpe)*sc:.4f}'),
                 (key, 'bin_amplitude_delta_nd_TNm', f'{abs(nd)*sc:.4f}'),
                 (key, 'bin_amplitude_f_b_TNm', f'{abs(fb)*sc:.4f}'),
                 (key, 'bin_phase_nd_minus_fb_deg',
                  f'{np.degrees(np.angle(nd / fb)):.1f}')]
        chk = np.sqrt(2 * P['$\\Delta N_D$'][kpk]) * U['$\\Delta N_D$'][1].std()
        assert abs(abs(nd) * sc - chk) < 1e-6, 'amplitude scaling disagrees'

        # the illusion, quantified: how often does a surrogate beat the
        # observed band correlation?
        bt = bandpass(*U['slab descent rate'])
        for nm in ('trench pull (stabilised)', '$\\Delta N_D$', '$F_B$'):
            tux2, y2 = U[nm]
            r = float(np.corrcoef(bandpass(tux2, y2), bt)[0, 1])
            nullr = np.array([np.corrcoef(bandpass(tux2, surrogate(y2, rng)),
                                          bt)[0, 1] for _ in range(N_SURR)])
            p = float((np.abs(nullr) >= abs(r)).mean())
            print(f'   band r(slab descent, {nm:26s}) = {r:+.2f}  p = {p:.3f}'
                  f'   [median |r| of surrogates {np.median(np.abs(nullr)):.2f}]')
            sl = nm.strip('$').replace(' ', '_').replace('\\', '') \
                   .replace('(', '').replace(')', '')
            rows += [(key, f'band_corr_slab_{sl}', f'{r:.3f}'),
                     (key, f'band_corr_p_slab_{sl}', f'{p:.4f}'),
                     (key, f'surrogate_median_abs_r_{sl}',
                      f'{np.median(np.abs(nullr)):.3f}')]

        for ax in axes[:, col]:
            ax.grid(alpha=0.25, color=C_RULE, lw=0.6)

    for row in (1, 2):
        hi = max(max(abs(v) for v in ax.get_ylim()) for ax in axes[row, :])
        for ax in axes[row, :]:
            ax.set_ylim(-hi, hi)

    fig.suptitle("Does the trench pull carry power at the slab's buckling "
                 "frequency? Per-bin, with an illusion control", fontsize=11.5)
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
                            'sampling': '37 snapshots at 2 Myr; bins 74/37/24.7/18.5/...',
                            'bins_in_band': 'TWO — 24.7 and 18.5 Myr; the per-bin '
                                            'table is the honest statement',
                            'surrogates': 'phase-randomised, amplitude spectrum preserved',
                            'slab_proxies': 'mean dip + upper-mantle descent rate; '
                                            'buckling interpretation inherited from '
                                            'the earlier paper (Dan 2026-09-23)',
                            'note': 'diagnostic, not a manuscript figure'})
    print('written:', tab)


if __name__ == '__main__':
    main()
