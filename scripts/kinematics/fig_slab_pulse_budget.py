"""fig_slab_pulse_budget — when the slab speeds up, who pays for it?

Writes figures/kinematics/fig_slab_pulse_budget.png and
tables/slab_pulse_budget.csv. Needs the column cache and the slab cache.

Dan's closing question, 2026-09-22: when the slab descent rate increases,
by what percentage does it typically increase, how much extra RESISTANCE
does that deliver through basal drag and tilting, and is that supplied by
an increase in the trench pull or by a decrease in Delta N_D?

METHOD. Every series is linearly detrended. Each balance term is
regressed on the detrended slab descent rate, and the slope is evaluated
at ONE TYPICAL FLUCTUATION -- one standard deviation of that series. So
every number below is "the force change that accompanies a typical
speed-up". The balance

    TP + NT - TILT = Delta N_D + F_B

must hold for those increments as well as for the values, and it does:
the budget closes to 0.001 TN/m in STD and 0.000 in WAL. That closure is
what makes the partition below a measurement rather than a fit.

  1  the force change per typical speed-up, term by term.
  2  the extra resistance (F_B + TILT), and how it is paid for.

THE ANSWER, AND THE TWO MODELS GIVE OPPOSITE ONES.

  STD  slab descent fluctuates +/-12 %, plate velocity responds +19 %.
       Extra resistance +0.342 TN/m (drag +0.255, tilt +0.087).
       Paid for by Delta N_D RELAXING: +0.259, or 76 % of it.
       Trench pull supplies 15 %, the thermal term 10 %.
       ⇒ THE DRIVING TERM DOES NOT CHANGE AT ALL: dGPE* moves -0.001
         TN/m, r = -0.01. The slab's signal reaches the plate purely as a
         reduction in resistance.

  WAL  slab descent fluctuates +/-23 %, plate velocity responds +32 %.
       Extra resistance +0.258 TN/m (drag +0.201, tilt +0.057).
       Paid for by the TRENCH PULL: +0.192, or 74 % of it.
       Delta N_D supplies 16 %.

⚠ In STD Delta N_D falls by 33 % of its mean (+0.78 -> +0.52) while
REMAINING COMPRESSIONAL. The slab does not need to reverse the sign of
the in-plane resultant to transmit a velocity change; it only needs to
perturb it. That is the point worth making.

⚠⚠ THE WAL PARTITION IS NOT ROBUST AND SHOULD NOT BE QUOTED. Recomputed
on subsets it swings wildly -- trench pull takes 17 % in the first half,
154 % in the second, 123 % on speed-ups alone and -4 % on slow-downs --
and the underlying regression barely exists: Delta N_D against the slab
descent rate has R2 = 0.02 in WAL. Fitting a slope through that is
meaningless, and the headline "74 % trench pull" is an artefact of the
second half and the 60-70 Myr event. WAL's panel is retained only to show
that the STD result is not universal.

THE STD PARTITION IS ROBUST. Recomputed the same ways, Delta N_D supplies
76 % (all points), 76 % (first half), 72 % (second half) and 79 %
(largest third of fluctuations only). The regression is sound: Delta N_D
against slab descent has R2 = 0.80 linear, and a quadratic adds only 0.02,
so a single slope is adequate.

  It IS asymmetric, which is worth knowing: Delta N_D supplies 54 % on
  speed-ups and 80 % on slow-downs, with the trench pull taking 31 % of
  speed-ups. The relaxation route dominates more when the slab slows than
  when it accelerates.

METHOD NOTE. "Typical" is a REGRESSION ACROSS THE WHOLE RECORD, not a
picked event: ordinary least squares on every snapshot, evaluated at one
standard deviation of the detrended descent rate. Because it is linear, a
2 s.d. event gives exactly twice the force change. Note also that linear
detrending removes only the secular trend, so the residual contains the
~22 Myr oscillation, any curvature AND the noise -- it is not
specifically short-wavelength.

DRAFT CAPTION. The force budget of a typical slab speed-up, for STD and
WAL. Each balance term is regressed on the slab descent rate, both
linearly detrended, and evaluated at one standard deviation of the
descent rate -- 12 % of its mean in STD and 23 % in WAL. Top: the
resulting change in each term. Bottom: the extra resistance delivered by
the velocity-slaved terms, basal drag and tilting, and the terms that
supply it. In STD the driving term does not change and the in-plane
resultant relaxes to pay for the extra resistance; in WAL the trench pull
supplies most of it.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE)); sys.path.insert(0, _HERE)
import column_profiles_cache as cpc
import slab_geometry_cache as sgc
from fig_budget_time import terms
from tables_io import write_table

ROOT = os.path.dirname(os.path.dirname(_HERE))
C = {'basal drag $F_B$': 'red', 'tilting': '#7B3294',
     'trench pull': '#0072B2', '$-\\Delta N_D$ (relaxing)': '0.35',
     'non-tilting (thermal)': '#D55E00'}


def main():
    if not hasattr(np, 'trapezoid'):
        np.trapezoid = np.trapz
    d, s = cpc.load(), sgc.load()
    ld = lambda t, y: y - np.polyval(np.polyfit(t, y, 1), t)
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.6),
                             gridspec_kw={'height_ratios': [1.35, 1.0]})
    rows = [('model', 'quantity', 'value')]

    for col, key in enumerate(('STD', 'WAL')):
        c = cpc.derive(d, key); q = terms(d, key)
        z, zc, t = c['z'], c['zc'], c['t']
        ts = s[f'{key}_t']; n = min(len(t), len(ts)); t = t[:n]
        vz = s[f'{key}_vz_upper'][:n]
        vp = np.abs(d[f'{key}_vx_TR'][:, c['z'] / 1e3 < 20].mean(axis=1))[:n]
        TP = -np.trapezoid(c['p_T'][:n, zc], z[zc], axis=1) / 1e12
        RP = np.trapezoid(c['p_R'][:n, zc], z[zc], axis=1) / 1e12
        NT = np.trapezoid(c['p_R_untilted'][:n, zc], z[zc], axis=1) / 1e12
        TILT = -c['dP'][:n] * cpc.ZC_KM * 1e3 / 1e12
        ND, FB = q['d_nd'][:n] / 1e12, q['f_b'][:n] / 1e12

        a_vz = ld(t, vz); sd = a_vz.std()
        sens = lambda y: np.polyfit(a_vz, ld(t, y), 1)[0] * sd
        dTP, dNT, dTILT = sens(TP), sens(NT), sens(TILT)
        dND, dFB, dGPE = sens(ND), sens(FB), sens(TP + RP)
        dvp = sens(vp)
        pct_slab = 100 * sd / vz.mean()
        pct_plate = 100 * dvp / vp.mean()
        resist = dFB + dTILT
        budget = dTP + dNT - dTILT - dND - dFB

        a0, a1 = axes[0, col], axes[1, col]
        items = [('basal drag $F_B$', dFB), ('tilting', dTILT),
                 ('trench pull', dTP), ('non-tilting (thermal)', dNT),
                 ('$-\\Delta N_D$ (relaxing)', -dND)]
        names = [n_ for n_, _ in items]
        vals = [v for _, v in items]
        a0.barh(names, vals, color=[C[n_] for n_ in names], height=0.6)
        a0.axvline(0, color='k', lw=1.0)
        a0.set_xlabel('force change per typical speed-up [TN/m]', fontsize=9.5)
        a0.set_title(f'{key}   slab +{pct_slab:.0f} %, plate +{pct_plate:.0f} %'
                     f'   (drive $\\Delta$GPE* {dGPE:+.3f})', fontsize=10.5)
        a0.tick_params(axis='y', labelsize=8.5)
        for i, v in enumerate(vals):
            a0.text(v + 0.006 * np.sign(v), i, f'{v:+.3f}', va='center',
                    ha='left' if v > 0 else 'right', fontsize=8)

        # bottom: how the extra resistance is paid for
        supply = [('trench pull', dTP), ('$-\\Delta N_D$ (relaxing)', -dND),
                  ('non-tilting (thermal)', dNT)]
        left = 0.0
        for nm, v in supply:
            a1.barh([0], [100 * v / resist], left=left, color=C[nm], height=0.5,
                    label=f'{nm}  ({100*v/resist:.0f} %)')
            a1.text(left + 50 * v / resist, 0, f'{100*v/resist:.0f} %',
                    ha='center', va='center', fontsize=9,
                    color='white' if abs(v / resist) > 0.3 else 'k')
            left += 100 * v / resist
        a1.set_yticks([]); a1.set_xlim(0, 105)
        a1.set_xlabel(f'share of the extra resistance ({resist:+.3f} TN/m: '
                      f'drag {dFB:+.3f} + tilt {dTILT:+.3f})', fontsize=9.5)
        a1.legend(frameon=False, fontsize=8, loc='lower center',
                  bbox_to_anchor=(0.5, -1.05), ncol=3)
        for ax in (a0, a1):
            ax.grid(alpha=0.25, color='#BFC3D1', lw=0.6, axis='x')

        print(f'{key}: slab +{pct_slab:.0f} %, plate +{pct_plate:.0f} %; '
              f'resistance {resist:+.3f} = drag {dFB:+.3f} + tilt {dTILT:+.3f}; '
              f'supplied by TP {dTP:+.3f} ({100*dTP/resist:.0f} %), '
              f'-dN_D {-dND:+.3f} ({-100*dND/resist:.0f} %), '
              f'NT {dNT:+.3f} ({100*dNT/resist:.0f} %); '
              f'drive {dGPE:+.3f}; budget closes to {budget:+.4f}')
        rows += [(key, 'slab_fluctuation_percent', f'{pct_slab:.1f}'),
                 (key, 'plate_response_percent', f'{pct_plate:.1f}'),
                 (key, 'd_basal_drag_TNm', f'{dFB:.4f}'),
                 (key, 'd_tilting_TNm', f'{dTILT:.4f}'),
                 (key, 'd_trench_pull_TNm', f'{dTP:.4f}'),
                 (key, 'd_non_tilting_TNm', f'{dNT:.4f}'),
                 (key, 'd_delta_nd_TNm', f'{dND:.4f}'),
                 (key, 'd_drive_TNm', f'{dGPE:.4f}'),
                 (key, 'extra_resistance_TNm', f'{resist:.4f}'),
                 (key, 'share_trench_pull_percent', f'{100*dTP/resist:.1f}'),
                 (key, 'share_nd_relaxing_percent', f'{-100*dND/resist:.1f}'),
                 (key, 'budget_closure_TNm', f'{budget:.4f}')]

    fig.suptitle('The force budget of a typical slab speed-up — who supplies '
                 'the extra resistance?', fontsize=11.5)
    fig.tight_layout()
    out = os.path.join(ROOT, 'figures', 'kinematics', 'fig_slab_pulse_budget.png')
    fig.savefig(out, bbox_inches='tight', dpi=220)
    print('written:', out)
    tab = write_table('slab_pulse_budget', rows[0], rows[1:],
                      script='kinematics/fig_slab_pulse_budget.py',
                      figure='kinematics/fig_slab_pulse_budget.png',
                      models=('STD', 'WAL'),
                      meta={'unit': 'one s.d. of the detrended slab descent rate',
                            'detrend': 'linear, per series',
                            'identity': 'TP + NT - TILT = dN_D + F_B'})
    print('written:', tab)


if __name__ == '__main__':
    main()
