"""fig_flexure_kinematics — V, M and the trench deflection against PLATE kinematics.

Writes figures/kinematics/fig_flexure_kinematics.png and
tables/flexure_kinematics.csv. Needs the column cache and the bending
caches (notebooks/further_analysis/outputs/bending_moment_{STD,WAL}.npz).

Requested by Dan 2026-09-22. fig_compliance_time established that the
deflection CAN be explained from V and M -- STD fits at (C_V, C_M) =
(1.47, 1.11), within a factor of two of elastic, with correlation in both
the trend and the residual. The question this figure asks is what sets V
and M themselves, looking at PLATE kinematics rather than slab descent
(fig_slab_correlations covers the slab side).

Dan's hypothesis: M might be larger where the rollback velocity is high.

  1  the search: correlation of each flexural quantity against each
     kinematic quantity, per model, all detrended.
  2  |V_T| against rollback and the partition fraction.
  3  |M_T| against plate velocity and convergence.

Rows 2 and 3 show the SAME pairings in both models, so the contrast
between them is readable rather than having to be asserted.

WHAT IT FINDS. The hypothesis is half right, and it is the other half
that is interesting.

  WAL  |V_T| is set by ROLLBACK, r = +0.85, and tracks the rollback
       FRACTION 1-f at +0.80 -- V is large when the trench is doing the
       moving rather than the plate.
       |M_T| is set by PLATE VELOCITY (+0.66) and convergence (+0.64).
       So the two flexural resultants answer to different halves of the
       kinematic partition.

  STD  none of it holds. |M_T| correlates with nothing (|r| <= 0.11) and
       |V_T| only weakly with convergence (+0.41).

That difference lines up with the compliance fit: in WAL the moment
carries the entire deflection (C_M = 2.66, C_V = -0.04) and the moment is
kinematically controlled, so the deflection follows plate velocity
(+0.65). In STD both resultants contribute and neither is kinematically
controlled.

Outlier rejection and the alpha solve follow fig_compliance_time: 3 MAD
about a 5-point running median on w_T, which removed 4 points in STD and
5 in WAL and was the difference between a meaningful fit and a
meaningless one.

DRAFT CAPTION. Flexural resultants and trench deflection against plate
kinematics, for STD and WAL, all series linearly detrended. Top:
correlation of the shear resultant, bending moment, measured deflection
and modelled deflection against plate velocity, trench rollback,
convergence rate and the fraction of convergence carried by plate motion.
Middle and bottom: the two pairings that carry the WAL signal, shown for
both models.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.ndimage import median_filter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE)); sys.path.insert(0, _HERE)
import column_profiles_cache as cpc
from fig_compliance_time import elastic_x0, DRHO, G, T_MIN
from tables_io import write_table

ROOT = os.path.dirname(os.path.dirname(_HERE))
C = {'plate velocity': '#1B9E77', 'rollback': '#E7298A',
     'convergence': 'k', 'rollback fraction $1-f$': '0.55'}
C_V, C_M = '#7B3294', '#D55E00'


def series(key):
    f = dict(np.load(os.path.join(ROOT, 'notebooks', 'further_analysis', 'outputs',
                                  f'bending_moment_{key}.npz')))
    t, w = f['t_myr'], f['w_T']
    V, M = np.abs(f['V_T']), np.abs(f['M'])
    xI, hnp = f['x_I'] - f['x_T'], f['h_np']
    ok = (w > 200) & (hnp < 59e3) & (t > T_MIN)
    al = np.full(len(t), np.nan)
    for k in np.where(ok)[0]:
        try:
            al[k] = brentq(lambda a: elastic_x0(V[k], M[k], a, DRHO, G) - xI[k],
                           5e3, 500e3)
        except Exception:
            pass
    ok &= np.isfinite(al)
    res = w - median_filter(w, size=5, mode='nearest')
    mad = np.median(np.abs(res - np.median(res))) * 1.4826
    ok &= np.abs(res) <= 3 * mad
    t, w, V, M, al = (x[ok] for x in (t, w, V, M, al))
    A = np.vstack([V * al, M]).T
    (cV, cM), *_ = np.linalg.lstsq(A, w * DRHO * G * al ** 2, rcond=None)
    wm = (cV * V * al + cM * M) / (DRHO * G * al ** 2)
    return t, dict(V=V, M=M, w=w, wm=wm), (cV, cM)


def main():
    d = cpc.load()
    fig, axes = plt.subplots(3, 2, figsize=(12.6, 10.4),
                             gridspec_kw={'height_ratios': [1.15, 1.0, 1.0]})
    rows = [('model', 'quantity', 'value')]
    ld = lambda t, y: y - np.polyval(np.polyfit(t, y, 1), t)
    zs = lambda y: y / y.std()

    for col, key in enumerate(('STD', 'WAL')):
        t, fx, (cV, cM) = series(key)
        c = cpc.derive(d, key); tc = c['t']; zkm = c['z'] / 1e3
        vp = np.abs(d[f'{key}_vx_TR'][:, zkm < 20].mean(axis=1))
        roll = np.gradient(d[f'{key}_xT'] / 1e3, tc) / 10.0
        conv = vp + roll
        kin = {'plate velocity': np.interp(t, tc, vp),
               'rollback': np.interp(t, tc, roll),
               'convergence': np.interp(t, tc, conv),
               # the ROLLBACK FRACTION rather than the plate fraction
               # (Dan, 2026-09-22): 1 - f is what V actually tracks, and
               # detrended it is exactly -f, so the correlation simply
               # changes sign while the quantity gains a physical name --
               # the share of convergence taken by the trench.
               'rollback fraction $1-f$': np.interp(t, tc, 1.0 - vp / conv)}
        flex = {'$|V_T|$': fx['V'], '$|M_T|$': fx['M'],
                '$w_T$ measured': fx['w'], '$w_T$ model': fx['wm']}

        Rm = np.array([[np.corrcoef(ld(t, fy), ld(t, ky))[0, 1]
                        for ky in kin.values()] for fy in flex.values()])
        a0 = axes[0, col]
        im = a0.imshow(Rm, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        a0.set_xticks(range(len(kin))); a0.set_xticklabels(kin, fontsize=8.5,
                                                          rotation=20, ha='right')
        a0.set_yticks(range(len(flex))); a0.set_yticklabels(flex, fontsize=9)
        for i in range(Rm.shape[0]):
            for j in range(Rm.shape[1]):
                a0.text(j, i, f'{Rm[i, j]:+.2f}', ha='center', va='center',
                        fontsize=9,
                        color='white' if abs(Rm[i, j]) > 0.55 else 'k')
        a0.set_title(f'{key}   ($C_V$ = {cV:.2f}, $C_M$ = {cM:.2f}, n = {len(t)})',
                     fontsize=10.5)

        a1 = axes[1, col]
        a1.plot(t, zs(ld(t, fx['V'])), '-', color=C_V, lw=2.2, label='$|V_T|$')
        for nm in ('rollback', 'rollback fraction $1-f$'):
            rr = np.corrcoef(ld(t, fx['V']), ld(t, kin[nm]))[0, 1]
            a1.plot(t, zs(ld(t, kin[nm])), '-', color=C[nm], lw=1.6,
                    label=f'{nm}  ({rr:+.2f})')
        a1.set_ylabel('$|V_T|$ and kinematics\n(detrended, normalised)', fontsize=8.5)

        a2 = axes[2, col]
        a2.plot(t, zs(ld(t, fx['M'])), '-', color=C_M, lw=2.2, label='$|M_T|$')
        for nm in ('plate velocity', 'convergence'):
            rr = np.corrcoef(ld(t, fx['M']), ld(t, kin[nm]))[0, 1]
            a2.plot(t, zs(ld(t, kin[nm])), '-', color=C[nm], lw=1.6,
                    label=f'{nm}  ({rr:+.2f})')
        a2.set_ylabel('$|M_T|$ and kinematics\n(detrended, normalised)', fontsize=8.5)
        a2.set_xlabel('Model time [Myr]', fontsize=11)
        for ax in (a1, a2):
            ax.axhline(0, color='k', lw=0.8)
            ax.legend(frameon=False, fontsize=8, loc='upper left', ncol=3)
            ax.grid(alpha=0.25, color='#BFC3D1', lw=0.6)

        print(f'{key} (n={len(t)}): ' + '; '.join(
            f'{fn} -> ' + ', '.join(f'{kn} {Rm[i, j]:+.2f}'
                                    for j, kn in enumerate(kin))
            for i, fn in enumerate(flex)))
        for i, fn in enumerate(flex):
            for j, kn in enumerate(kin):
                sl = lambda x: ''.join(ch for ch in x.lower().replace(' ', '_')
                                       if ch.isalnum() or ch == '_')
                rows.append((key, f'corr_{sl(fn)}_{sl(kn)}', f'{Rm[i, j]:.3f}'))

    fig.colorbar(im, ax=axes[0, :].tolist(), fraction=0.03, pad=0.02,
                 label='detrended $r$')
    fig.suptitle('What sets $V$ and $M$? Flexural resultants against plate '
                 'kinematics', fontsize=11.5)
    out = os.path.join(ROOT, 'figures', 'kinematics', 'fig_flexure_kinematics.png')
    fig.savefig(out, bbox_inches='tight', dpi=220)
    print('written:', out)
    tab = write_table('flexure_kinematics', rows[0], rows[1:],
                      script='kinematics/fig_flexure_kinematics.py',
                      figure='kinematics/fig_flexure_kinematics.png',
                      models=('STD', 'WAL'),
                      meta={'detrend': 'linear, per series',
                            'outliers': '3 MAD about a 5-point median on w_T',
                            'kinematics': 'interpolated from the column cache'})
    print('written:', tab)


if __name__ == '__main__':
    main()
