"""fig_trench_resultants — N_D, V and M at the trench through the runs.

Writes figures/fig_trench_resultants.png from the committed caches:
notebooks/outputs/time_evolution_{STD,WAL}.npz (N_D, V at x_T; legacy Fd
naming) and notebooks/further_analysis/outputs/bending_moment_{...}.npz
(M at x_T, local-pivot h_np; bending-notebook implementation).

Styling per FIGURE_STYLE.md (house style from the time-evolution
notebook's series figures): models overlaid, STD navy #002147 / WAL
magenta #E5007D, solid with 'o' markers and white marker edges, rule
grid; one quantity per panel, three stacked panels sharing model time.

Register note on V (SYMBOLOGY §7.5): the plotted V is the notebook's
extracted V = ∫τ_zx dz at the trench column — the repo's engineering
sense, which is the NEGATIVE of the companion register's V. Any prose
quoting a V relation must say which V it means.

DRAFT CAPTION. The resultants at the trench column through the runs:
the normal-stress-difference resultant N_D (top), the vertical shear
resultant V (middle), and the bending moment M about the neutral plane
(bottom), for STD (navy) and WAL (magenta). N_D is predominantly
compression-like; V and M carry the trench topography (the vertical
load and moment support of the deflection).
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T_MIN_MYR = 8.0
C_STD, C_WAL, C_RULE = '#002147', '#E5007D', '#BFC3D1'

def main():
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for key, color in (('STD', C_STD), ('WAL', C_WAL)):
        te = np.load(os.path.join(ROOT, 'notebooks', 'outputs',
                                  f'time_evolution_{key}.npz'))
        t = te['t_yr'] / 1e6
        m = t >= T_MIN_MYR
        bm = np.load(os.path.join(ROOT, 'notebooks', 'further_analysis',
                                  'outputs', f'bending_moment_{key}.npz'))
        tb = bm['t_myr']
        mb = (tb >= T_MIN_MYR) & np.isfinite(bm['M'])
        style = dict(color=color, lw=2.0, ms=5.5, markeredgecolor='white',
                     markeredgewidth=0.6)
        axes[0].plot(t[m], te['Fd_xT'][m] * 1e-12, '-o', label=key, **style)
        axes[1].plot(t[m], te['V_xT'][m] * 1e-12, '-o', label=key, **style)
        axes[2].plot(tb[mb], bm['M'][mb] * 1e-17, '-o', label=key, **style)
        print(f'{key}: N_D(x_T) median {np.median(te["Fd_xT"][m])/1e12:+.2f} TN/m; '
              f'V(x_T) median {np.median(te["V_xT"][m])/1e12:+.2f} TN/m; '
              f'M(x_T) median {np.median(bm["M"][mb])/1e17:+.2f} x10^17 N '
              f'({mb.sum()} steps)')
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
