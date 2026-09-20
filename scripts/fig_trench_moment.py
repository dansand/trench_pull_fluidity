"""fig_trench_moment — the bending moment at the trench through time (SI).

Writes figures/fig_trench_moment.png from the committed column-profile
cache, via column_profiles_cache.resultants().

Moved out of the main-text resultants figure 2026-09-21 (Dan: the bending
moment is the less important of the trench resultants and belongs in the
supplement). It is retained because it is what makes "bending moment" a
measured rather than an assumed quantity, and because the neutral-plane
depth it is taken about is the pivot used elsewhere.

M is taken about the trench neutral plane, itself the extremum of the
cumulative fibre stress over 5-60 km; the pivot sensitivity is dM/dz =
-N_D, so the neutral-plane depth is plotted alongside.

DRAFT CAPTION. The bending moment at the trench column through the runs
for STD (navy) and WAL (magenta), taken about the local neutral plane
(lower panel), whose depth is the extremum of the cumulative fibre
stress.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import column_profiles_cache as cpc
from tables_io import write_table

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T_MIN_MYR = 8.0
C = {'STD': '#002147', 'WAL': '#E5007D'}
C_RULE = '#BFC3D1'

def main():
    d = cpc.load()
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.4), sharex=True,
                             gridspec_kw={'height_ratios': [2, 1]})
    rows = [('model', 'quantity', 'value')]
    for key in ('STD', 'WAL'):
        col = C[key]
        r = cpc.resultants(d, key)
        m = r['t'] >= T_MIN_MYR
        axes[0].plot(r['t'][m], r['m_T'][m] * 1e-17, '-o', color=col, lw=1.9,
                     ms=4, markeredgecolor='white', markeredgewidth=0.5, label=key)
        axes[1].plot(r['t'][m], r['h_np'][m] / 1e3, '-', color=col, lw=1.9)
        print(f'{key}: M(x_T) median {np.median(r["m_T"][m])/1e17:+.2f} x10^17 N; '
              f'h_np median {np.median(r["h_np"][m])/1e3:.0f} km')
        rows += [(key, 'moment_trench_median_1e17N', f'{np.median(r["m_T"][m])/1e17:.3f}'),
                 (key, 'moment_trench_q1_1e17N', f'{np.percentile(r["m_T"][m], 25)/1e17:.3f}'),
                 (key, 'moment_trench_q3_1e17N', f'{np.percentile(r["m_T"][m], 75)/1e17:.3f}'),
                 (key, 'neutral_plane_median_km', f'{np.median(r["h_np"][m])/1e3:.1f}')]
    axes[0].set_ylabel(r'$M$ at the trench [$10^{17}$ N]', fontsize=11)
    axes[1].set_ylabel('neutral plane\n$h_{np}$ [km]', fontsize=10)
    axes[1].set_xlabel('Model time [Myr]', fontsize=12)
    axes[1].invert_yaxis()
    for ax in axes:
        ax.grid(alpha=0.25, color=C_RULE, lw=0.6)
    axes[0].axhline(0, color='k', lw=1.2)
    axes[0].legend(frameon=False, fontsize=10)
    fig.tight_layout()
    out = os.path.join(ROOT, 'figures', 'fig_trench_moment.png')
    fig.savefig(out, bbox_inches='tight', dpi=220)
    print('written:', out)
    tab = write_table('trench_moment', rows[0], rows[1:],
                      script='fig_trench_moment.py', figure='fig_trench_moment.png',
                      models=('STD', 'WAL'),
                      meta={'pivot': 'trench neutral plane (extremum of cumulative '
                                     'fibre stress, 5-60 km)', 't_min_Myr': T_MIN_MYR})
    print('written:', tab)

if __name__ == '__main__':
    main()
