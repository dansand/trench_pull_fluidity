"""fig_trench_resultants — N_D, V and M at the trench through the runs.

Writes figures/fig_trench_resultants.png from the committed
COLUMN-PROFILE cache, via the shared column_profiles_cache.resultants().
It previously read two notebook-built caches (time-evolution for N_D and
V, bending-moment for M); moved 2026-09-17 so that every resultant in
the paper comes from one cache, one set of pickers and one integration
depth. N_D and V reproduce the notebook values to better than 1 %; M
differs by ~10 % because the pivot is now the trench neutral plane from
the same cumulative fibre stress the rest of the pipeline uses.

Styling per FIGURE_STYLE.md (house style from the time-evolution
notebook's series figures): models overlaid, STD navy #002147 / WAL
magenta #E5007D, solid with 'o' markers and white marker edges, rule
grid; one quantity per panel, three stacked panels sharing model time.

Register note on V (SYMBOLOGY §7.5): the plotted V is the extracted
V = ∫τ_zx dz at the trench column — the repo's engineering
sense, which is the NEGATIVE of the companion register's V. Any prose
quoting a V relation must say which V it means.

DRAFT CAPTION. The resultants at the trench column through the runs:
the normal-stress-difference resultant N_D (top), the vertical shear
resultant V (middle), and the bending moment M about the neutral plane
(bottom), for STD (navy) and WAL (magenta). N_D is predominantly
compression-like; V and M carry the trench topography (the vertical
load and moment support of the deflection).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import column_profiles_cache as cpc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T_MIN_MYR = 8.0
C_STD, C_WAL, C_RULE = '#002147', '#E5007D', '#BFC3D1'

def main():
    d = cpc.load()
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for key, color in (('STD', C_STD), ('WAL', C_WAL)):
        r = cpc.resultants(d, key)
        t = r['t']
        m = t >= T_MIN_MYR
        style = dict(color=color, lw=2.0, ms=5.5, markeredgecolor='white',
                     markeredgewidth=0.6)
        axes[0].plot(t[m], r['nd_T'][m] * 1e-12, '-o', label=key, **style)
        axes[1].plot(t[m], r['v_T'][m] * 1e-12, '-o', label=key, **style)
        axes[2].plot(t[m], r['m_T'][m] * 1e-17, '-o', label=key, **style)
        print(f'{key}: N_D(x_T) median {np.median(r["nd_T"][m])/1e12:+.2f} TN/m; '
              f'V(x_T) median {np.median(r["v_T"][m])/1e12:+.2f} TN/m; '
              f'M(x_T) median {np.median(r["m_T"][m])/1e17:+.2f} x10^17 N; '
              f'h_np median {np.median(r["h_np"][m])/1e3:.0f} km ({m.sum()} steps)')
    axes[0].set_ylabel(r'$N_D(x_T)$' + '\nForce per unit distance [TN/m]',
                       fontsize=12)
    axes[1].set_ylabel(r'$V(x_T)$' + '\nForce per unit distance [TN/m]',
                       fontsize=12)
    axes[2].set_ylabel(r'$M_T$ [$10^{17}$ N]', fontsize=12)
    axes[2].set_xlabel('Model time [Myr]', fontsize=13)
    for ax in axes:
        ax.axhline(0, color='k', lw=0.7)
        ax.grid(True, alpha=0.25, color=C_RULE, lw=0.6)
        ax.tick_params(labelsize=10)
    axes[0].legend(loc='best', frameon=False, fontsize=11)
    axes[0].set_title('Resultants at the trench column', fontsize=13)
    fig.tight_layout()
    out = os.path.join(ROOT, 'figures', 'fig_trench_resultants.png')
    fig.savefig(out, bbox_inches='tight', dpi=220)
    print('written:', out)

if __name__ == '__main__':
    main()
