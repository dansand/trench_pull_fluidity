"""fig_nd_trench_ridge — N_D at the trench and ridge columns, and their
difference, through the run.

Writes figures/fig_nd_trench_ridge.png from the committed time-evolution
caches (notebooks/outputs/time_evolution_{STD,WAL}.npz; cache keys carry
the legacy Fd naming for N_D).

The setup this figure makes explicit (Dan, 2026-09-15): N_D at the ridge
is always tension-like but SMALL, so the plate-wide difference
N_D(x_T) − N_D(x_R) is approximately the trench value whatever its sign —
pulling or resisting. This unstated approximation is what underlies the
conventional slab-pull framework: any edge force at the trench is
effectively the change in N_D across the plate. Measured, the trench
value is predominantly compression-like, so the difference predominantly
resists.

Axis reading: for the two RESULTANTS, positive = tension, negative =
compression (sign-convention statement). For the DIFFERENCE, positive =
driving (a net tension-like edge force pulling the trailing plate
trench-ward), negative = resisting. The plate-wide difference is written
out explicitly as N_D(x_T) − N_D(x_R); the register's bare Δ operator is
reserved for the (x_I) − (x_T) column difference (SYMBOLOGY §5).

Run statistics printed on execution (t ≥ 8 Myr; committed-cache sourced):
N_D(x_T) median −0.70 TN/m, tension-like 37 % of steps (STD); −1.89 TN/m,
0 % (WAL). N_D(x_R) median +0.39 / +0.24 TN/m, positive at every step.

Resultant vs net force (Dan, 2026-09-15): N_D at a single column is a
COLUMNWISE RESULTANT — the depth integral of the normal-stress difference
on one vertical plane, with no force meaning until paired with that
plane's outward normal. The difference between two columns IS a net
force: N_D(x_T) − N_D(x_R) is the net force per unit distance that the
two end planes exert on the intervening plate segment.

DRAFT CAPTION. The normal-stress-difference resultant N_D at the trench
column (black) and ridge column (grey), and their difference
N_D(x_T) − N_D(x_R) (dashed), through the runs. Each single-column value
is a columnwise resultant; the difference is the net force per unit
distance exerted on the plate segment between the two columns. N_D at
the ridge is tension-like at every step but small, so the difference is
approximately the trench value whatever its sign — the implicit reading
of the conventional slab-pull framework, in which a tension-like edge
force at the trench is the plate-wide change in N_D. The measured trench
value is predominantly compression-like: the edge term mostly resists,
and the driving force must be carried by the topographic pressure
gradient.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T_MIN_MYR = 8.0

def main():
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2), sharey=True, sharex=True)
    for ax, k in zip(axes, ('STD', 'WAL')):
        d = np.load(os.path.join(ROOT, 'notebooks', 'outputs', f'time_evolution_{k}.npz'))
        t = d['t_yr'] / 1e6
        m = t >= T_MIN_MYR
        nd_t = d['Fd_xT'][m] / 1e12
        nd_r = (d['Fd_xT'][m] + d['delta_Fd_R'][m]) / 1e12   # N_D(x_R)
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

if __name__ == '__main__':
    main()
