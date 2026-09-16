"""fig_ridge_push_decomposition — the three areas that make up ridge push.

Writes figures/fig_ridge_push_decomposition.png from the committed
column-profile cache. One panel per model (STD | WAL), mid-run average.

The construction (Dan, 2026-09-16), all from one depth rule:

  solid   the sigma_zz anomaly of the ridge column relative to x_I, AS THE
          MODEL PRODUCES IT. Its sign change, z*, is the integration depth
          (equivalently the maximum of its cumulative integral).
  dashed  the same profile shifted by the deep asthenospheric value
          |Delta P| so its deep asymptote is zero — the column with the
          back-tilt of the whole plate removed.

  area 1  DYNAMIC RIDGE PUSH — under the measured curve to z*. This is what
          enters the force balance.
  area 2  BACK-TILT — the strip between the two curves over 0..z*, equal to
          |Delta P| * z*. Even above z*, the measured anomaly already
          contains this contribution from the tilting of the entire plate:
          correcting for it makes ridge push LARGER.
  area 3  the density-structure TAIL below z*, which completes the
          STATIC RIDGE PUSH (Dan's term): what the density structure alone
          would supply if it carried no dynamics — no transition from a
          lithospheric convective boundary layer to asthenospheric
          counterflow. Read from the PLATEAU of the cumulative integral of
          the shifted profile, never by integrating to its zero crossing:
          the shifted profile is flat where it approaches zero, so the
          crossing is ill-conditioned while the integral is bounded
          (plateau spread 0.1-0.3 %).

Mid-run values [TN/m]: dynamic 1.35 / 1.47, back-tilt 0.65 / 0.45,
static 2.21 / 2.03 (STD / WAL). Single implementation of the partition:
column_profiles_cache.partition().

DRAFT CAPTION. Decomposition of the ridge-to-first-isostatic-column
vertical normal stress anomaly, mid-run average, for STD (left) and WAL
(right). The solid curve is the anomaly as the model produces it; the
dashed curve is the same profile with the plate-wide back-tilt removed
(shifted by the asthenospheric value Delta P). Dark shading is the
dynamic ridge push that enters the force balance, integrated to the depth
z* at which the anomaly changes sign; mid shading is the back-tilt
contribution contained within it; light shading is the remaining density
structure below z*. The dynamic and static ridge push differ by the
back-tilt plus that tail.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import column_profiles_cache as cpc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = {'STD': '#002147', 'WAL': '#E5007D'}
C_RULE = '#BFC3D1'

def main():
    d = cpc.load()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 6.0), sharey=True, sharex=True)
    for ax, key in zip(axes, ('STD', 'WAL')):
        col = C[key]
        c = cpc.derive(d, key)
        p = cpc.partition(d, key)
        m = c['mid']
        z, zkm = c['z'], c['z'] / 1e3
        meas = gaussian_filter1d(c['p_R'][m].mean(axis=0), 2) / 1e6        # MPa
        dP = c['dP'][m].mean() / 1e6
        shift = meas - dP                                                   # untilted
        zs = p['z_star'][m].mean() / 1e3
        above = zkm <= zs
        ax.fill_betweenx(zkm[above], 0, meas[above], color=col, alpha=0.55, lw=0,
                         label='dynamic ridge push (in the balance)')
        ax.fill_betweenx(zkm[above], meas[above], shift[above], color=col, alpha=0.28,
                         lw=0, label=r'back-tilt  $|\Delta P|\,z^{*}$')
        below = (zkm >= zs)
        ax.fill_betweenx(zkm[below], 0, shift[below], color=col, alpha=0.10, lw=0,
                         label='density-structure tail below $z^{*}$')
        ax.plot(meas, zkm, color=col, lw=2.0, label='as modelled')
        ax.plot(shift, zkm, color=col, lw=1.4, ls='--', label='back-tilt removed')
        ax.axhline(zs, color='k', lw=0.9, ls=':')
        ax.axvline(0, color='k', lw=1.6)
        ax.axvline(dP, color='0.5', lw=1.0, ls='-.')
        ax.text(dP, 235, r'$\Delta P$', fontsize=9, color='0.35', ha='center')
        ax.text(1.0, zs - 4, r'$z^{*}$', fontsize=10, ha='left', va='bottom')
        dyn, tilt, static = (p['measured'][m].mean() / 1e12,
                             p['tilt'][m].mean() / 1e12,
                             p['static'][m].mean() / 1e12)
        ax.set_title(f'{key}:  dynamic {dyn:.2f}  +  back-tilt {tilt:.2f}  '
                     f'$\\rightarrow$  static {static:.2f} TN/m', fontsize=10)
        ax.set_xlabel(r'$\sigma_{zz}$ anomaly, ridge $-\,x_I$ [MPa]', fontsize=11)
        ax.grid(alpha=0.2, color=C_RULE, lw=0.6)
        print(f'{key}: z* {zs:.0f} km, dP {dP:+.2f} MPa | dynamic {dyn:+.2f}, '
              f'back-tilt {tilt:+.2f}, tail {p["tail"][m].mean()/1e12:+.2f}, '
              f'static {static:+.2f} TN/m')
    axes[0].set_ylabel('Depth [km]', fontsize=11)
    axes[0].set_ylim(250, 0)
    axes[0].set_xlim(-12, 46)
    axes[0].legend(frameon=False, fontsize=8, loc='lower right')
    fig.suptitle('Ridge push, decomposed: what the balance contains, what the '
                 'back-tilt removes, and the static limit', fontsize=11)
    fig.tight_layout()
    out = os.path.join(ROOT, 'figures', 'fig_ridge_push_decomposition.png')
    fig.savefig(out, bbox_inches='tight', dpi=220)
    print('written:', out)

if __name__ == '__main__':
    main()
