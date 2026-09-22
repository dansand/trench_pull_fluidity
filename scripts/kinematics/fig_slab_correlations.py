"""fig_slab_correlations — what tracks the slab's descent rate.

Writes figures/kinematics/fig_slab_correlations.png and
tables/slab_correlations.csv. Needs both caches
(scripts/column_profiles_cache.py, scripts/kinematics/slab_geometry_cache.py).

Requested by Dan 2026-09-22, extending fig_slab_trench_coupling from
three quantities to the six that plausibly respond to slab descent. Seven
curves on one axis would be unreadable, so they are GROUPED by what kind
of quantity they are, with the slab velocity repeated faintly in each
group as the common reference:

  1  KINEMATICS      plate velocity and convergence
  2  LOAD -> TOPOGRAPHY   the shear-supported depth w_tau = (dV/dx)/rho g
                     and the measured trench depth
  3  FORCE RESPONSE  trench pull and Delta N_D at the trench
  4  SUMMARY         every correlation as a bar, against BOTH the slab
                     descent rate and the tip advance rate

⚠ V ITSELF IS NOT THE RIGHT VARIABLE, and an earlier version of this
figure plotted it. The flexure relation is w_tau = (dV/dx)/rho g -- the
topography is set by the GRADIENT of the shear resultant, not by the
resultant. Measured:

    r(trench depth, V)      = -0.12 (STD) / -0.09 (WAL)   <- nothing
    r(trench depth, dV/dx)  = +0.89        / +0.88        <- everything

and dV/dx reproduces the trench depth to 95 % (1830 m against a measured
1925 m in STD). Plotting -V produced a spurious-looking anti-correlation
with the slab velocity; the sign was right and the variable was wrong.

The bending moment is dropped for now (Dan, 2026-09-22).

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
from scipy.signal import savgol_filter

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
     '$V(x_T)$': '#7B3294',
     '$M(x_T)$': '#D55E00', 'trench depth': '#E7298A',
     'trench pull': '#0072B2', '$-\\Delta N_D$': '0.35'}


def main():
    if not hasattr(np, 'trapezoid'):
        np.trapezoid = np.trapz
    d, s = cpc.load(), sgc.load()
    ld = lambda t, y: y - np.polyval(np.polyfit(t, y, 1), t)
    zs = lambda y: y / y.std()
    # NO sharex: row 3 is a bar chart whose x axis is a correlation, not
    # time. Sharing it with the time-series rows drives their limits to
    # +/-1 and pushes every curve off-screen.
    fig, axes = plt.subplots(4, 2, figsize=(13.0, 12.6),
                             gridspec_kw={'height_ratios': [1.1, 1.0, 1.0, 1.0]})
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

        Vt = r['v_T'][:n] / 1e12               # V as extracted (see docstring)
        Mt = r['m_T'][:n] / 1e17               # bending moment at the trench
        TP = -np.trapezoid(c['p_T'][:n, zc], z[zc], axis=1) / 1e12

        # LIGHT ZERO-PHASE SMOOTHING for display (Dan, 2026-09-22): trench
        # pull in particular is noisy enough to hide its own signal. A
        # 5-point Savitzky-Golay is symmetric, so it injects no lag. It
        # DOES raise correlations by removing noise, so the table carries
        # both the smoothed and the raw value.
        sm = lambda y: savgol_filter(y, 5, 2)
        a_vz = sm(ld(tt, vz))
        raw = {}

        grp = [
            ('kinematics', [('plate velocity', ld(tt, vp)),
                            ('convergence', ld(tt, conv))]),
            ('flexure $\\rightarrow$ topography',
             [('$V(x_T)$', ld(tt, Vt)), ('$M(x_T)$', ld(tt, Mt)),
              ('trench depth', ld(tt, wT))]),
            ('force response', [('trench pull', ld(tt, TP)),
                                ('$-\\Delta N_D$', ld(tt, -ND))]),
        ]
        for ax, (lab, items) in zip(axes[:3, col], grp):
            ax.plot(tt, zs(a_vz), '-', color=C_SLAB, lw=2.8, alpha=0.30,
                    label='slab $v_z$ (reference)')
            for nm, y in items:
                ys = sm(y)
                rr = float(np.corrcoef(a_vz, ys)[0, 1])
                raw[nm] = (rr, float(np.corrcoef(ld(tt, vz), y)[0, 1]))
                ax.plot(tt, zs(ys), '-', color=C[nm], lw=1.7,
                        label=f'{nm}  ({rr:+.2f})')
            ax.axhline(0, color='k', lw=0.8)
            ax.set_ylabel(f'{lab}\n(detrended, smoothed)', fontsize=8.5)
            ax.legend(frameon=False, fontsize=8, loc='upper left', ncol=2)
            ax.grid(alpha=0.25, color='#BFC3D1', lw=0.6)
        axes[0, col].set_title(key, fontsize=11)
        for ax in axes[:2, col]:
            ax.set_xlim(axes[0, col].get_xlim()); ax.tick_params(labelbottom=False)
        axes[2, col].set_xlim(axes[0, col].get_xlim())
        axes[2, col].set_xlabel('Model time [Myr]', fontsize=10.5)

        names = [nm for _, items in grp for nm, _ in items]
        vals = [raw[nm][0] for nm in names]
        a3 = axes[3, col]
        order = np.argsort(vals)
        a3.barh([names[i] for i in order], [vals[i] for i in order],
                color=[C[names[i]] for i in order], height=0.62)
        a3.axvline(0, color='k', lw=1.0); a3.set_xlim(-1, 1)
        a3.set_xlabel('detrended $r$ with slab $v_z$ (smoothed)', fontsize=10)
        a3.tick_params(axis='y', labelsize=8.5)
        a3.grid(alpha=0.25, color='#BFC3D1', lw=0.6, axis='x')

        print(f'{key}: ' + ', '.join(f'{nm} {raw[nm][0]:+.2f}'
                                     f'({raw[nm][1]:+.2f} raw)' for nm in names))
        for nm in names:
            slug = ''.join(ch for ch in nm.lower().replace(' ', '_')
                           if ch.isalnum() or ch == '_')
            rows += [(key, f'corr_slabvz_{slug}_smoothed', f'{raw[nm][0]:.3f}'),
                     (key, f'corr_slabvz_{slug}_raw', f'{raw[nm][1]:.3f}')]

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
