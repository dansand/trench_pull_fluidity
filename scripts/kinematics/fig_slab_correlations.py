"""fig_slab_correlations — what tracks the slab's descent rate.

Writes figures/kinematics/fig_slab_correlations.png and
tables/slab_correlations.csv. Needs both caches
(scripts/column_profiles_cache.py, scripts/kinematics/slab_geometry_cache.py).

Requested by Dan 2026-09-22, extending fig_slab_trench_coupling from
three quantities to the six that plausibly respond to slab descent. Seven
curves on one axis would be unreadable, so they are GROUPED by what kind
of quantity they are, with the slab velocity repeated faintly in each
group as the common reference:

  1  KINEMATICS      trailing-plate velocity, convergence rate
  2  TRENCH LOAD     trench depth, the downward shear load -V, and the
                     bending moment M at the trench
  3  SUMMARY         every correlation against the slab descent rate as a
                     bar, so all six are comparable at a glance

Delta N_D sits in the summary rather than in a time panel: it already has
its own figure (fig_slab_trench_coupling) where it is plotted against the
slab velocity directly.

EVERYTHING IS LINEARLY DETRENDED AND THEN DIVIDED BY ITS OWN S.D. These
quantities carry different units and very different secular trends; at
levels they would all correlate above 0.9 simply by rising together. See
fig_slab_descent's docstring for what that mistake looks like when it is
not caught.

SIGN NOTE. The plotted shear resultant is -V, the downward load on the
trailing plate's trench-side face, matching fig_trench_resultants. The
extracted V is its negative and is also the negative of the companion
register's V (SYMBOLOGY §7.5).

⚠ WAL's correlations average across a regime change. Its slab/trench
coupling holds to ~50 Myr and then fails, in the window where the trench
advances (fig_slab_trench_coupling). Single WAL numbers from this figure
therefore average a real relationship with its own breakdown; read them
with that figure beside them.

DRAFT CAPTION. Quantities that respond to the slab's descent rate, for
STD (left) and WAL (right), each linearly detrended and divided by its
standard deviation. Top: the trailing-plate velocity and the convergence
rate, with the slab's upper-mantle descent rate repeated faintly for
reference. Middle: the trench depth, the downward shear load on the
trench-side face, and the bending moment at the trench. Bottom:
correlation of each quantity with the slab descent rate.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))          # the shared scripts/ dir
sys.path.insert(0, _HERE)
import column_profiles_cache as cpc
import slab_geometry_cache as sgc
from fig_budget_time import terms
from tables_io import write_table

ROOT = os.path.dirname(os.path.dirname(_HERE))
RHO_G = 3300.0 * 9.8
C_SLAB = '#0072B2'
C = {'plate velocity': '#1B9E77', 'convergence': 'k',
     'trench depth': '#E7298A', '$-V$ (trench face load)': '#7B3294',
     'bending moment $M$': '#D55E00', '$\\Delta N_D$': '0.35'}


def main():
    if not hasattr(np, 'trapezoid'):
        np.trapezoid = np.trapz
    d, s = cpc.load(), sgc.load()
    ld = lambda t, y: y - np.polyval(np.polyfit(t, y, 1), t)
    zs = lambda y: y / y.std()
    # NO sharex: row 3 is a bar chart whose x axis is a correlation, not
    # time. Sharing it with the time-series rows drives their limits to
    # +/-1 and pushes every curve off-screen.
    fig, axes = plt.subplots(3, 2, figsize=(12.6, 10.0),
                             gridspec_kw={'height_ratios': [1.2, 1.2, 0.95]})
    rows = [('model', 'quantity', 'value')]

    for col, key in enumerate(('STD', 'WAL')):
        c = cpc.derive(d, key); q = terms(d, key); r = cpc.resultants(d, key)
        t, z, zc = c['t'], c['z'], c['zc']
        ts = s[f'{key}_t']
        n = min(len(t), len(ts))
        assert np.allclose(t[:n], ts[:n], atol=0.3), f'{key}: time axes differ'
        tt = t[:n]
        vz = s[f'{key}_vz_upper'][:n]
        vp = np.abs(d[f'{key}_vx_TR'][:, c['z'] / 1e3 < 20].mean(axis=1))[:n]
        conv = vp + (np.gradient(d[f'{key}_xT'] / 1e3, t) / 10.0)[:n]
        s0 = c['p_T'][:n, 0] - 0.5 * (c['p_T'][:n, 1] - c['p_T'][:n, 0])
        wT = -s0 / RHO_G
        mV = -r['v_T'][:n] / 1e12              # downward trench-face load
        mM = r['m_T'][:n] / 1e17               # bending moment
        ND = q['d_nd'][:n] / 1e12

        a_vz = ld(tt, vz)
        grp1 = [('plate velocity', ld(tt, vp)), ('convergence', ld(tt, conv))]
        grp2 = [('trench depth', ld(tt, wT)),
                ('$-V$ (trench face load)', ld(tt, mV)),
                ('bending moment $M$', ld(tt, mM))]
        extra = [('$\\Delta N_D$', ld(tt, ND))]

        a0, a1, a2 = (axes[row, col] for row in range(3))
        for ax, grp, lab in ((a0, grp1, 'kinematics'), (a1, grp2, 'trench load')):
            ax.plot(tt, zs(a_vz), '-', color=C_SLAB, lw=2.6, alpha=0.35,
                    label='slab $v_z$ (reference)')
            for nm, y in grp:
                rr = float(np.corrcoef(a_vz, y)[0, 1])
                ax.plot(tt, zs(y), '-', color=C[nm], lw=1.7,
                        label=f'{nm}  ($r$ = {rr:+.2f})')
            ax.axhline(0, color='k', lw=0.8)
            ax.set_ylabel(f'{lab}\n(detrended, normalised)', fontsize=9)
            ax.legend(frameon=False, fontsize=8, loc='upper left', ncol=2)
            ax.grid(alpha=0.25, color='#BFC3D1', lw=0.6)
        a0.set_title(key, fontsize=11)
        a1.set_xlim(a0.get_xlim())
        a0.tick_params(labelbottom=False)
        a1.set_xlabel('Model time [Myr]', fontsize=10.5)

        allq = grp1 + grp2 + extra
        names = [nm for nm, _ in allq]
        rr = [float(np.corrcoef(a_vz, y)[0, 1]) for _, y in allq]
        order = np.argsort(rr)
        a2.barh([names[i] for i in order], [rr[i] for i in order],
                color=[C[names[i]] for i in order], height=0.62)
        a2.axvline(0, color='k', lw=1.0)
        a2.set_xlim(-1, 1)
        a2.set_xlabel('$r$ with slab descent rate (detrended)', fontsize=10)
        a2.tick_params(axis='y', labelsize=8.5)
        a2.grid(alpha=0.25, color='#BFC3D1', lw=0.6, axis='x')

        print(f'{key}: ' + ', '.join(f'{nm} {v:+.2f}' for nm, v in zip(names, rr)))
        for nm, v in zip(names, rr):
            slug = (nm.replace('$', '').replace('\\Delta ', 'delta_')
                      .replace(' ', '_').replace('(', '').replace(')', '')
                      .replace('-', 'minus').lower())
            rows.append((key, f'corr_slab_vz_{slug}', f'{v:.3f}'))

    fig.suptitle('What tracks the slab descent rate — all quantities detrended '
                 'and normalised', fontsize=11.5)
    fig.tight_layout()
    out = os.path.join(ROOT, 'figures', 'kinematics', 'fig_slab_correlations.png')
    fig.savefig(out, bbox_inches='tight', dpi=220)
    print('written:', out)
    tab = write_table('slab_correlations', rows[0], rows[1:],
                      script='kinematics/fig_slab_correlations.py',
                      figure='kinematics/fig_slab_correlations.png',
                      models=('STD', 'WAL'),
                      meta={'normalisation': 'linear detrend, then / s.d.',
                            'reference': 'slab v_z, 200-600 km',
                            'V_sense': '-V, downward load on the trench-side face'})
    print('written:', tab)


if __name__ == '__main__':
    main()
