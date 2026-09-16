"""fig_force_vs_depth — choosing the integration depth for the
trailing-plate part of the balance.

Writes figures/fig_force_vs_depth.png from the committed column-profile
cache (column_profiles_cache.py).

The problem it settles (Dan, 2026-09-16): the trench pull converges with
depth, but the x_I-to-ridge part of the sigma_zz balance does not — it
carries the non-converging asthenospheric (tilt) contribution, so the
quoted number appears to depend on an arbitrary integration depth. The
figure shows it does not, in the depth window that matters:

  left  — cumulative trench pull (solid) and cumulative x_I-to-ridge part
          (dashed) as functions of integration depth
  right — their ratio, which is FLAT across 60–100 km

Ratio at the mid-run reference: STD 0.65 (60 km) / 0.66 (75 km) / 0.65
(crossing); WAL 0.85 / 0.87 / 0.87. Integrating to the per-snapshot
sigma_zz zero crossing and to a fixed z_c = 75 km agree within ~2 %
(run medians: ridge push +1.47 vs +1.45 STD, +1.54 vs +1.54 WAL TN/m).
The zero crossing is the physically justified rule (base of the
coherently translating plate — see fig_lab_kinematics); the insensitivity
is what makes the number quotable.

DRAFT CAPTION. Cumulative forces per unit distance as a function of the
depth of integration, mid-run average, for STD (navy) and WAL (magenta).
Left: the trench pull (solid) converges by about 60 km, while the
first-isostatic-column-to-ridge part (dashed) peaks near the vertical
normal stress zero crossing (dotted) and then declines as the
asthenospheric pressure gradient takes over. Right: their ratio is flat
through the depth range in which the cumulative maximum falls, so the
partition between trench pull and the plate-wide term does not depend
materially on where the integration is stopped.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import column_profiles_cache as cpc
from fig_lab_kinematics import crossing_depth

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = {'STD': '#002147', 'WAL': '#E5007D'}
C_RULE = '#BFC3D1'
# The shaded band is the RUN RANGE (IQR) of the depth at which the
# cumulative x_I-to-ridge curve is a MAXIMUM — i.e. of the depth where the
# integrand changes sign. It is computed, not chosen: an earlier fixed
# 60-100 km band was an eyeballed guess and is retired (Dan, 2026-09-16).

def main():
    d = cpc.load()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.8), sharey=True)
    for key in ('STD', 'WAL'):
        col = C[key]
        c = cpc.derive(d, key)
        m = c['mid']
        z, zkm = c['z'], c['z'] / 1e3
        p_t = gaussian_filter1d(c['p_T'][m].mean(axis=0), 2)
        p_r = gaussian_filter1d(c['p_R'][m].mean(axis=0), 2)
        cum = lambda a: np.concatenate([[0.0], np.cumsum(0.5 * (a[1:] + a[:-1]) * np.diff(z))])
        TP = -cum(p_t) / 1e12
        RP = cum(p_r) / 1e12
        z_cross = crossing_depth(p_r / 1e6, zkm)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = RP / TP
        axes[0].plot(TP, zkm, color=col, lw=2.0, label=f'{key}   trench pull')
        axes[0].plot(RP, zkm, color=col, lw=1.5, ls='--', label=f'{key}   $x_I$ to ridge')
        axes[1].plot(ratio, zkm, color=col, lw=2.0, label=key)
        # depth of the cumulative maximum, per snapshot (reported, not shaded —
        # the shaded band was unreadable, Dan 2026-09-16)
        sel = (zkm > 30) & (zkm < 200)
        zmax_all = [zkm[sel][int(np.argmax(cum(gaussian_filter1d(pr, 2))[sel]))]
                    for pr in c['p_R']]
        for ax in axes:
            ax.axhline(z_cross, color=col, lw=0.9, ls=':')
        print(f'   depth of cumulative maximum: median {np.median(zmax_all):.0f} km, '
              f'IQR {np.percentile(zmax_all, 25):.0f}-{np.percentile(zmax_all, 75):.0f}, '
              f'full {min(zmax_all):.0f}-{max(zmax_all):.0f}')
        at = lambda a, zz: float(np.interp(zz, zkm, a))
        print(f'{key}: trench pull {at(TP, 75):.2f} TN/m (60 km {at(TP, 60):.2f}, '
              f'100 km {at(TP, 100):.2f}); x_I-to-ridge {at(RP, 75):.2f} '
              f'(crossing {z_cross:.0f} km: {at(RP, z_cross):.2f}); '
              f'ratio {at(ratio, 60):.2f}/{at(ratio, 75):.2f}/{at(ratio, z_cross):.2f} '
              f'(60 / 75 / crossing)')
    for ax in axes:
        ax.axvline(0, color='k', lw=1.4)
        ax.grid(alpha=0.2, color=C_RULE, lw=0.6)
    axes[0].set_xlabel('Cumulative force per unit distance [TN/m]', fontsize=11)
    axes[1].set_xlabel(r'ratio   ($x_I$ to ridge) / trench pull', fontsize=11)
    axes[1].set_xlim(0, 1.4)
    axes[0].set_ylabel('Depth of integration [km]', fontsize=11)
    axes[0].set_ylim(200, 0)
    axes[0].legend(frameon=False, fontsize=9, loc='lower right')
    axes[1].legend(frameon=False, fontsize=10)
    fig.suptitle('Choosing the depth of integration: cumulative forces and their ratio\n'
                 '(dotted: $\\sigma_{zz}$ sign change at mid-run — the maximum of the cumulative curve)', fontsize=11)
    fig.tight_layout()
    out = os.path.join(ROOT, 'figures', 'fig_force_vs_depth.png')
    fig.savefig(out, bbox_inches='tight', dpi=220)
    print('written:', out)

if __name__ == '__main__':
    main()
