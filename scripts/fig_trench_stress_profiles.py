"""fig_trench_stress_profiles — the two stress profiles at the column of
maximum bending moment, with the mechanical-thickness estimates marked.

Writes figures/fig_trench_stress_profiles.png from the committed
column-profile cache. Drawn at the reference epoch, averaged over the
mid-run window (36-44 Myr, conventions §4b) rather than a single
snapshot: the profiles are noisy at the trench and the average is the
stable representation.

Two panels, side by side, both at the column of MAXIMUM BENDING MOMENT
(Dan, 2026-09-16) — ~20-25 km seaward of the trench, where the flexural
signal is strongest:
  left   the NORMAL-STRESS DIFFERENCE profile, sigma_xx - sigma_zz — the
         flexural fibre stress. Its zero crossing is the neutral plane;
         the outer-fibre peaks bound the strong part of the plate.
  right  the DELTA SIGMA_ZZ profile, that column minus the first isostatic
         column (pressure register) — the topographic pressure deficit.

Overlaid on both, per model: the span of the four independent mechanical
thickness estimates (2 h_np, yield-10 %, yield-50 MPa, thermal 900 C) as
a shaded band, with the neutral plane h_np marked separately. Reading the
two panels together is the point — the same thickness has to make sense
of the flexural envelope on the left and of the pressure deficit on the
right, and it does: the deficit dies where the envelope does.

DRAFT CAPTION. Stress profiles at the column of maximum bending moment
(about 20-25 km seaward of the trench), averaged over the mid-run window,
for STD (navy) and WAL (magenta). Left: the normal-stress difference
(flexural fibre stress); the neutral plane (dotted) is its zero crossing.
Right: the vertical normal stress difference between that column and the
first isostatic column. Shaded bands span the four independent estimates
of the mechanical thickness; both the flexural envelope and the pressure
deficit decay away within the same depth range.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import column_profiles_cache as cpc
from fig_mechanical_thickness import thicknesses

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = {'STD': '#002147', 'WAL': '#E5007D'}
C_RULE = '#BFC3D1'

def main():
    d = cpc.load()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 6.4), sharey=True)
    for key in ('STD', 'WAL'):
        col = C[key]
        c = cpc.derive(d, key)
        r = thicknesses(d, key)
        m = c['mid']
        z, zkm = c['z'], c['z'] / 1e3
        sm = lambda a: gaussian_filter1d(a, 2)
        fib = sm((d[f'{key}_sxx_M'] - d[f'{key}_szz_M'])[m].mean(axis=0)) / 1e6
        dzz = -sm((d[f'{key}_szz_M'] - d[f'{key}_szz_I'])[m].mean(axis=0)) / 1e6
        dx_M = (d[f'{key}_xM'][m] - d[f'{key}_xT'][m]).mean() / 1e3
        axes[0].plot(fib, zkm, color=col, lw=2.0, label=key)
        axes[1].plot(dzz, zkm, color=col, lw=2.0, label=key)
        ests = np.array([r[k][m].mean() for k in ('np2', 'yield10', 'yield50',
                                                  'thermal')]) / 1e3
        h_np = r['np2'][m].mean() / 2e3
        for ax in axes:
            ax.axhspan(ests.min(), ests.max(), color=col, alpha=0.10, lw=0)
            ax.axhline(np.median(ests), color=col, lw=1.0, ls='--')
        axes[0].axhline(h_np, color=col, lw=1.0, ls=':')
        axes[0].text(0.97, h_np - 1.5, f'$h_{{np}}$ {h_np:.0f} km', color=col,
                     fontsize=8, ha='right', transform=axes[0].get_yaxis_transform())
        print(f'{key}: max-|M| column at x_T{dx_M:+.0f} km; h_np {h_np:.0f} km; estimates {ests.min():.0f}-{ests.max():.0f} km '
              f'(median {np.median(ests):.0f}); fibre peaks '
              f'{fib[zkm < h_np].max():+.0f} / {fib[(zkm > h_np) & (zkm < 120)].min():+.0f} MPa; '
              f'deficit peak {dzz.min():+.0f} MPa')
    axes[0].set_xlabel(r'$\sigma_{xx}-\sigma_{zz}$ [MPa]', fontsize=11)
    axes[1].set_xlabel(r'$\Delta\sigma_{zz}$ relative to $x_I$ [MPa]'
                       '\n(pressure register)', fontsize=11)
    axes[0].set_ylabel('Depth [km]', fontsize=11)
    axes[0].set_ylim(120, 0)
    for ax in axes:
        ax.axvline(0, color='k', lw=1.6)
        ax.grid(alpha=0.25, color=C_RULE, lw=0.6)
        ax.legend(frameon=False, fontsize=10, loc='lower right')
    fig.suptitle('At the maximum-bending-moment column, mid-run average: the flexural '
                 'envelope and the\npressure deficit, with the span of '
                 'mechanical-thickness estimates (shaded) and the neutral plane (dotted)',
                 fontsize=10.5)
    fig.tight_layout()
    out = os.path.join(ROOT, 'figures', 'fig_trench_stress_profiles.png')
    fig.savefig(out, bbox_inches='tight', dpi=220)
    print('written:', out)

if __name__ == '__main__':
    main()
