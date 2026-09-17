"""fig_force_length_scale — the force a topographic perturbation delivers,
expressed as a length scale ("force admittance").

Writes figures/fig_force_length_scale.png and tables/force_length_scale.csv
from the committed column-profile cache.

The idea (Dan, 2026-09-16), by analogy with gravitational admittance: a
gravitational admittance says how much gravity signal a metre of
topography produces. Here we ask how much FORCE a metre of topography
produces. Because the force is a stress resultant and the surface load is
a stress, their ratio is a LENGTH:

    L = (force per unit distance) / (surface stress anomaly)
      = int (Delta sigma_zz) dz / (Delta rho g w)

so L is the depth over which the surface load would have to act at full
amplitude to deliver the measured force. It is the same quantity as the
"conversion rate" of the Earth-budget discussion divided by Delta rho g.

Three length scales, per snapshot:
  trench          trench pull / (trench pressure deficit at the surface).
                  Identically half the triangle-equivalent thickness, so
                  this is a direct restatement of the trench-pull scaling.
  ridge, dynamic  the x_I-to-ridge term that enters the force balance,
                  divided by the ridge surface anomaly
  ridge, static   the same for the static ridge push — COMPUTED AND
                  TABULATED but deliberately not plotted: it distracts
                  from the near-identity of the two measured conversions

The comparison is the point: the trench's non-isostatic deficit and the
plate-wide isostatic topography convert to driving force at almost
exactly the same rate — about 33 km, i.e. 1.07-1.08 GN/m per metre of
relief — despite arising by completely different mechanisms.

DRAFT CAPTION. The force delivered per unit of surface topography,
expressed as a length scale, through the runs for STD (navy) and WAL
(magenta): the trench (solid) and the ridge term that enters the force balance
(dashed). A larger length scale means topography that converts to driving
force more efficiently; the two track each other closely throughout.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import column_profiles_cache as cpc
from tables_io import write_table

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = {'STD': '#002147', 'WAL': '#E5007D'}
C_RULE = '#BFC3D1'
RHO_G = 3300.0 * 9.8          # Delta rho g (no ocean in these models)

def main():
    d = cpc.load()
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    rows = [('model', 'quantity', 'value')]
    for key in ('STD', 'WAL'):
        col = C[key]
        c = cpc.derive(d, key)
        p = cpc.partition(d, key)
        z = c['z']
        shallow = z <= 20e3
        sm = lambda a: gaussian_filter1d(a, 2)
        # surface stress anomalies = Delta rho g x (surface relief)
        s_T = np.array([np.abs(sm(a)[shallow]).max() for a in c['p_T']])
        s_R = np.array([sm(a)[shallow].max() for a in c['p_R']])
        cum = lambda a: np.concatenate([[0.0], np.cumsum(0.5 * (a[1:] + a[:-1]) * np.diff(z))])
        tp = np.array([np.interp(zs, z, -cum(sm(a)))
                       for a, zs in zip(c['p_T'], p['z_star'])])
        L_T = tp / s_T / 1e3
        L_Rd = p['measured'] / s_R / 1e3
        L_Rs = p['static'] / s_R / 1e3
        t = p['t']
        ax.plot(t, L_T, '-o', color=col, lw=2.0, ms=4.5, markeredgecolor='white',
                markeredgewidth=0.5, label=f'{key}  trench')
        ax.plot(t, L_Rd, '--', color=col, lw=1.6, label=f'{key}  ridge (in the balance)')
        # the static line is NOT plotted (Dan, 2026-09-16): it distracts from
        # the near-identity of the two measured conversions. Still tabulated.
        print(f'{key}: length scale [km] — trench {np.median(L_T):.1f} '
              f'(IQR {np.percentile(L_T,25):.0f}–{np.percentile(L_T,75):.0f}); '
              f'ridge dynamic {np.median(L_Rd):.1f}; ridge static {np.median(L_Rs):.1f}')
        print(f'   implied relief [m]: trench {np.median(s_T)/RHO_G:.0f}, '
              f'ridge {np.median(s_R)/RHO_G:.0f}')
        print(f'   conversion rate [GN/m per metre]: trench {RHO_G*np.median(L_T)*1e3/1e9:.2f}, '
              f'ridge dynamic {RHO_G*np.median(L_Rd)*1e3/1e9:.2f}, '
              f'ridge static {RHO_G*np.median(L_Rs)*1e3/1e9:.2f}')
        for name, arr in (('L_trench_km', L_T), ('L_ridge_dynamic_km', L_Rd),
                          ('L_ridge_static_km', L_Rs),
                          ('relief_trench_m', s_T / RHO_G),
                          ('relief_ridge_m', s_R / RHO_G)):
            rows += [(key, f'{name}_median', f'{np.median(arr):.2f}'),
                     (key, f'{name}_q1', f'{np.percentile(arr, 25):.2f}'),
                     (key, f'{name}_q3', f'{np.percentile(arr, 75):.2f}')]
    ax.set_xlabel('Model time [Myr]', fontsize=12)
    ax.set_ylabel('force per unit topography, as a length [km]', fontsize=12)
    ax.grid(alpha=0.25, color=C_RULE, lw=0.6)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    ax.set_title('How efficiently topography converts to driving force:\n'
                 'trench (non-isostatic) against ridge (isostatically compensated)',
                 fontsize=11)
    fig.tight_layout()
    out = os.path.join(ROOT, 'figures', 'fig_force_length_scale.png')
    fig.savefig(out, bbox_inches='tight', dpi=220)
    print('written:', out)
    tab = write_table('force_length_scale', rows[0], rows[1:],
                      script='fig_force_length_scale.py', figure='fig_force_length_scale.png',
                      models=('STD', 'WAL'), meta={'rho_g': RHO_G, 'surface_window_km': 20})
    print('written:', tab)

if __name__ == '__main__':
    main()
