"""RETIRED 2026-09-21 — consolidated into fig_trench_resultants.

Dan folded this figure into the resultants figure: panel (b) there now
carries N_D at the trench, first isostatic column and ridge. Kept as
source history; not part of the figure set and not run by reproduce.
Its table, tables/nd_trench_ridge.csv, is superseded by
tables/trench_resultants.csv.
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

def main():
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2), sharey=True, sharex=True)
    d = cpc.load()
    for ax, k in zip(axes, ('STD', 'WAL')):
        r = cpc.resultants(d, k)
        t = r['t']
        m = t >= T_MIN_MYR
        nd_t = r['nd_T'][m] / 1e12
        nd_r = r['nd_R'][m] / 1e12
        diff = nd_t - nd_r                                    # N_D(x_T) - N_D(x_R)
        ax.axhline(0, color='0.75', lw=0.8)
        ax.plot(t[m], nd_t, 'k-', lw=1.6, label='$N_D$ at the trench')
        ax.plot(t[m], nd_r, '-', color='0.55', lw=1.4, label='$N_D$ at the ridge')
        ax.plot(t[m], diff, 'k--', lw=1.2, label='$N_D(x_T) - N_D(x_R)$')
        ax.set_title(k, fontsize=10)
        ax.set_xlabel('t [Myr]')
        print(f'{k}: N_D(x_T) median {np.median(nd_t):+.2f} TN/m, '
              f'tension-like {(nd_t > 0).mean():.0%} of steps; '
              f'N_D(x_R) median {np.median(nd_r):+.2f}, always positive: {(nd_r > 0).all()}; '
              f'difference median {np.median(diff):+.2f}, driving {(diff > 0).mean():.0%} of steps')
    axes[0].set_ylabel('Force per unit distance [TN/m]')
    axes[0].legend(fontsize=8, frameon=False, loc='lower left')
    # axis reading: resultants (left edge) / difference (right edge)
    axes[0].text(0.02, 0.97, 'tension', transform=axes[0].transAxes,
                 fontsize=8, color='0.45', va='top')
    axes[0].text(0.02, 0.03, 'compression', transform=axes[0].transAxes,
                 fontsize=8, color='0.45', va='bottom')
    axes[1].text(0.98, 0.97, 'driving', transform=axes[1].transAxes,
                 fontsize=8, color='0.45', va='top', ha='right', style='italic')
    axes[1].text(0.98, 0.03, 'resisting', transform=axes[1].transAxes,
                 fontsize=8, color='0.45', va='bottom', ha='right', style='italic')
    fig.suptitle('$N_D$ at the trench and ridge columns, and the plate-wide difference',
                 fontsize=10.5)
    fig.tight_layout()
    out = os.path.join(ROOT, 'figures', 'fig_nd_trench_ridge.png')
    fig.savefig(out, dpi=200)
    print('written:', out)
    write_stats()

def write_stats():
    """Verifiable-numbers table for this figure (tables/nd_trench_ridge.csv).

    The quotable statistics: how large N_D at the ridge ever gets, and that
    maximum as a fraction of the average and maximum magnitudes of N_D at
    the trench — per model and pooled. All values TN/m unless 'fraction'
    or 'ratio'; t >= T_MIN_MYR throughout.
    """
    rows = [('model', 'quantity', 'value')]
    pool_r, pool_t = [], []
    d = cpc.load()
    for k in ('STD', 'WAL'):
        r = cpc.resultants(d, k)
        p = cpc.partition(d, k)
        m = r['t'] >= T_MIN_MYR
        nd_t = r['nd_T'][m] / 1e12
        nd_r = r['nd_R'][m] / 1e12
        diff = nd_t - nd_r
        gpe_diff = p['measured'][m] / 1e12
        pool_r.append(nd_r); pool_t.append(nd_t)
        rows += [
            (k, 'nd_ridge_max', f'{nd_r.max():.3f}'),
            (k, 'nd_ridge_median', f'{np.median(nd_r):.3f}'),
            (k, 'nd_ridge_always_tension_like', str(bool((nd_r > 0).all()))),
            (k, 'nd_trench_median', f'{np.median(nd_t):.3f}'),
            (k, 'nd_trench_mean_abs', f'{np.abs(nd_t).mean():.3f}'),
            (k, 'nd_trench_max_abs', f'{np.abs(nd_t).max():.3f}'),
            (k, 'nd_trench_tension_like_fraction', f'{(nd_t > 0).mean():.3f}'),
            (k, 'ratio_ridge_max_over_trench_mean_abs', f'{nd_r.max()/np.abs(nd_t).mean():.3f}'),
            (k, 'ratio_ridge_max_over_trench_max_abs', f'{nd_r.max()/np.abs(nd_t).max():.3f}'),
            (k, 'nd_diff_trench_minus_ridge_median', f'{np.median(diff):.3f}'),
            (k, 'nd_diff_driving_fraction', f'{(diff > 0).mean():.3f}'),
            (k, 'ratio_nd_diff_over_gpe_diff_max', f'{(np.abs(diff)/np.abs(gpe_diff)).max():.3f}'),
        ]
    pr, pt = np.concatenate(pool_r), np.concatenate(pool_t)
    rows += [
        ('POOLED', 'nd_ridge_max', f'{pr.max():.3f}'),
        ('POOLED', 'ratio_ridge_max_over_trench_mean_abs', f'{pr.max()/np.abs(pt).mean():.3f}'),
        ('POOLED', 'ratio_ridge_max_over_trench_max_abs', f'{pr.max()/np.abs(pt).max():.3f}'),
    ]
    tab = write_table('nd_trench_ridge', rows[0], rows[1:],
                      script='fig_nd_trench_ridge.py', figure='fig_nd_trench_ridge.png',
                      models=('STD', 'WAL'), meta={'t_min_Myr': T_MIN_MYR, 'zc_km': cpc.ZC_KM, 'ridge_pick': 'flow (max surface divergence)'})
    print('written:', tab)

if __name__ == '__main__':
    main()
