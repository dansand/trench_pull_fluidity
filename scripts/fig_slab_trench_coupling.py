"""fig_slab_trench_coupling — slab descent, trench depth and the in-plane resultant.

Writes figures/fig_slab_trench_coupling.png and
tables/slab_trench_coupling.csv. Needs both caches:
scripts/column_profiles_cache.py and scripts/slab_geometry_cache.py.

Requested by Dan 2026-09-22 after fig_slab_descent showed the
Delta N_D / slab-velocity relationship. Three quantities that should move
together if the trench is where the slab's inability to sink fast enough
is registered:

    slab v_z (200-600 km)     how fast material is actually leaving
    trench depth              the load that descent imposes on the plate
    -Delta N_D                minus the in-plane resultant, so that MORE
                              COMPRESSION plots DOWNWARD like the others

All three are LINEARLY DETRENDED and then standardised (divided by their
own standard deviation), because they carry different units and very
different secular trends. Without detrending they would all correlate at
>0.9 simply by rising together — see fig_slab_descent's docstring for
what that mistake looks like.

  1  the three normalised anomalies, with pairwise correlations.
  2  RUNNING correlation over a 22 Myr window, which is what shows that
     the relationship is not uniform in time.

WHAT IT ESTABLISHES.

STD: the coupling is strong and UNBROKEN. r(v_z, -Delta N_D) = +0.89
overall, and the running value stays between +0.85 and +0.96 for the
whole run. Slower descent, more compression, every cycle.

WAL: the same coupling holds from ~25 to ~50 Myr (running r reaches
+1.00 at 42 Myr) and then FAILS, falling to -0.35 by 58 Myr. Overall
r is +0.13, which averages a real relationship with its own breakdown and
should not be quoted on its own.

THE BREAKDOWN COINCIDES WITH TRENCH ADVANCE (shaded). In the same window
the trench rollback goes negative — the trench moves toward the
overriding plate instead of retreating — and the usual anti-correlation
between trench pull and ridge push REVERSES, reaching +0.58 and +0.71
where it is normally -0.85 or lower.

Dan's reading (2026-09-22, hypothesis, NOT established here): this is a
phase where the subduction zone partly breaks down — convergence slows,
the slab couples to both plates and pulls them down together, and the
trench advance reflects overriding-plate shortening, followed by a
release when convergence re-establishes. Testing it needs overriding-plate
diagnostics that this analysis does not have.

Colours are this figure's own family, not the domain or balance palettes:
slab velocity blue, trench depth magenta, -Delta N_D black.

DRAFT CAPTION. Coupling between slab descent, trench depth and the
in-plane resultant, for STD (left) and WAL (right). Top: each quantity
linearly detrended and divided by its standard deviation, with the
in-plane resultant negated so that greater compression plots downward.
Bottom: running correlations over a 22 Myr window, with intervals of
trench advance shaded. In STD the coupling between slab descent and the
in-plane resultant holds throughout; in WAL it holds until about 50 Myr
and then fails, in the same interval in which the trench advances and the
normally opposed trench pull and ridge push become positively correlated.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import column_profiles_cache as cpc
import slab_geometry_cache as sgc
from fig_budget_time import terms
from tables_io import write_table

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RHO_G = 3300.0 * 9.8
C_VZ = '#0072B2'
C_WT = '#E7298A'
C_ND = 'k'
C_RULE = '#BFC3D1'
WIN = 11                     # samples; 2 Myr cadence -> 22 Myr window


def main():
    if not hasattr(np, 'trapezoid'):
        np.trapezoid = np.trapz
    d, s = cpc.load(), sgc.load()
    ld = lambda t, y: y - np.polyval(np.polyfit(t, y, 1), t)
    zs = lambda y: y / y.std()
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.0), sharex=True,
                             gridspec_kw={'height_ratios': [1.5, 1.0]})
    rows = [('model', 'quantity', 'value')]

    for col, key in enumerate(('STD', 'WAL')):
        c = cpc.derive(d, key); q = terms(d, key)
        t, z, zc = c['t'], c['z'], c['zc']
        ts = s[f'{key}_t']
        n = min(len(t), len(ts))
        assert np.allclose(t[:n], ts[:n], atol=0.3), f'{key}: time axes differ'
        t = t[:n]
        vz = s[f'{key}_vz_upper'][:n]
        ND = q['d_nd'][:n] / 1e12
        s0 = c['p_T'][:n, 0] - 0.5 * (c['p_T'][:n, 1] - c['p_T'][:n, 0])
        wT = -s0 / RHO_G                                  # equivalent topography
        TP = -np.trapezoid(c['p_T'][:n, zc], z[zc], axis=1) / 1e12
        RP = np.trapezoid(c['p_R'][:n, zc], z[zc], axis=1) / 1e12
        roll = (np.gradient(d[f'{key}_xT'] / 1e3, c['t']) / 10.0)[:n]

        a_vz, a_nd, a_wt = ld(t, vz), ld(t, ND), ld(t, wT)
        a_tp, a_rp = ld(t, TP), ld(t, RP)
        r_vz_nd = float(np.corrcoef(a_vz, -a_nd)[0, 1])
        r_vz_wt = float(np.corrcoef(a_vz, a_wt)[0, 1])
        r_wt_nd = float(np.corrcoef(a_wt, -a_nd)[0, 1])

        a0, a1 = axes[0, col], axes[1, col]
        a0.plot(t, zs(a_vz), '-', color=C_VZ, lw=2.0,
                label=f'slab $v_z$, 200–600 km')
        a0.plot(t, zs(a_wt), '-', color=C_WT, lw=1.7, label='trench depth')
        a0.plot(t, zs(-a_nd), '-', color=C_ND, lw=1.7,
                label=r'$-\Delta N_D$  (down = more compression)')
        a0.axhline(0, color='k', lw=0.8)
        a0.set_ylabel('detrended, normalised\nby its own s.d.', fontsize=9.5)
        a0.set_title(key, fontsize=11)
        a0.legend(frameon=False, fontsize=8, loc='upper left', ncol=1)
        a0.text(0.98, 0.04,
                f'$r$($v_z$, $-\\Delta N_D$) = {r_vz_nd:+.2f}\n'
                f'$r$($v_z$, depth) = {r_vz_wt:+.2f}\n'
                f'$r$(depth, $-\\Delta N_D$) = {r_wt_nd:+.2f}',
                transform=a0.transAxes, fontsize=8, ha='right', va='bottom',
                color='0.25')

        run = lambda x_, y_: np.array(
            [np.corrcoef(x_[i:i+WIN], y_[i:i+WIN])[0, 1] for i in range(len(t)-WIN+1)])
        tc = t[WIN // 2: WIN // 2 + len(run(a_vz, -a_nd))]
        a1.plot(tc, run(a_vz, -a_nd), '-', color=C_VZ, lw=2.0,
                label=r'$v_z$ vs $-\Delta N_D$')
        a1.plot(tc, run(a_tp, a_rp), '--', color='0.35', lw=1.6,
                label='trench pull vs ridge push')
        a1.axhline(0, color='k', lw=0.9)
        a1.set_ylim(-1.05, 1.05)
        a1.set_ylabel(f'running $r$\n({WIN*2:.0f} Myr window)', fontsize=9.5)
        a1.set_xlabel('Model time [Myr]', fontsize=11)
        a1.legend(frameon=False, fontsize=8, loc='lower left')

        # shade intervals of TRENCH ADVANCE (rollback < 0) on both panels
        adv = roll < 0
        if adv.any():
            for ax in (a0, a1):
                ax.fill_between(t, *ax.get_ylim(), where=adv, color='0.75',
                                alpha=0.30, lw=0, zorder=0)
            a1.annotate('trench advances', xy=(t[adv].mean(), -0.85),
                        fontsize=8, color='0.35', ha='center')
        for ax in (a0, a1):
            ax.grid(alpha=0.25, color=C_RULE, lw=0.6)

        rr = run(a_vz, -a_nd)
        print(f'{key}: r(vz,-ND) {r_vz_nd:+.2f}, r(vz,depth) {r_vz_wt:+.2f}, '
              f'r(depth,-ND) {r_wt_nd:+.2f}; running r range '
              f'{rr.min():+.2f}..{rr.max():+.2f}; trench advances '
              f'{100*adv.mean():.0f} % of steps'
              + (f' ({t[adv].min():.0f}-{t[adv].max():.0f} Myr)' if adv.any() else ''))
        rows += [(key, 'corr_vz_minusND', f'{r_vz_nd:.3f}'),
                 (key, 'corr_vz_trenchdepth', f'{r_vz_wt:.3f}'),
                 (key, 'corr_trenchdepth_minusND', f'{r_wt_nd:.3f}'),
                 (key, 'running_corr_min', f'{rr.min():.3f}'),
                 (key, 'running_corr_max', f'{rr.max():.3f}'),
                 (key, 'corr_trenchpull_ridgepush',
                  f'{np.corrcoef(a_tp, a_rp)[0,1]:.3f}'),
                 (key, 'trench_advance_fraction_of_steps', f'{adv.mean():.3f}'),
                 (key, 'window_Myr', f'{WIN*2:.0f}')]

    fig.suptitle('Slab descent, trench depth and the in-plane resultant — '
                 'coupled throughout in STD, breaking down in WAL', fontsize=11.5)
    fig.tight_layout()
    out = os.path.join(ROOT, 'figures', 'fig_slab_trench_coupling.png')
    fig.savefig(out, bbox_inches='tight', dpi=220)
    print('written:', out)
    tab = write_table('slab_trench_coupling', rows[0], rows[1:],
                      script='fig_slab_trench_coupling.py',
                      figure='fig_slab_trench_coupling.png',
                      models=('STD', 'WAL'),
                      meta={'normalisation': 'linear detrend, then divided by s.d.',
                            'trench_depth': 'equivalent topography, '
                                            '-sigma_zz(0)/rho_g relative to x_I',
                            'shading': 'trench advance (rollback < 0)'})
    print('written:', tab)


if __name__ == '__main__':
    main()
