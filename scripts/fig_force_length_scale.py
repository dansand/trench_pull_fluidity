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
expressed as a length scale, through the runs for (a) STD and (b) WAL:
the trench-side term (blue) and the ridge-side term that enters the force
balance (orange), in the domain colours of the schematic. A larger length
scale means topography that converts to driving force more efficiently.
The two converge within the first ~30 Myr and track each other closely
thereafter, at about 33 km -- 1.07-1.08 GN/m per metre of relief --
despite arising by completely different mechanisms.
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
# DOMAIN COLOURS, not model colours (Dan, 2026-10-15). The repo carries two
# conventions and they do not conflict: colour means MODEL where both models
# share an axes (fig_mechanical_thickness, the preceding figure), and DOMAIN
# where the panel is one model (fig_column_anomalies_cumulative,
# fig_gpe_partition, fig_ridge_push_anatomy). This figure was the one trying
# to do both at once -- model in the colour, domain in the line style -- so
# it is split by model and the colour is freed for the domain.
C_TRENCH = '#0072B2'        # non-isostatic / trench pull domain (schematic)
C_RIDGE = '#D55E00'         # isostatic / ridge push domain (schematic)
C_RULE = '#BFC3D1'
RHO_G = 3300.0 * 9.8          # Delta rho g (no ocean in these models)

def main():
    d = cpc.load()
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.8), sharex=True,
                             sharey=True)
    rows = [('model', 'quantity', 'value')]
    for row, key in enumerate(('STD', 'WAL')):
        ax = axes[row]
        c = cpc.derive(d, key)
        p = cpc.partition(d, key)
        z = c['z']
        shallow = z <= 20e3
        sm = lambda a: gaussian_filter1d(a, 2)
        # surface stress anomalies = Delta rho g x (surface relief)
        s_T = np.array([np.abs(sm(a)[shallow]).max() for a in c['p_T']])
        s_R = np.array([sm(a)[shallow].max() for a in c['p_R']])
        # cell rule on cell edges, so a curve's endpoint equals the scalar
        cum = lambda a: np.concatenate(
            [[0.0], np.cumsum(a) * float(np.diff(z)[0])])[:len(z)]
        # TRENCH PULL TO THE FIXED z_c, like everything else the paper
        # reports (Dan, 2026-10-09). This integrated to each snapshot's own
        # crossing z*, which is why the repo carried two trench pulls.
        tp = -cpc.integrate_to_zc(np.array([sm(a) for a in c['p_T']]),
                                  z, axis=1)   # cell rule, 0 -> z_c
        L_T = tp / s_T / 1e3
        L_Rd = p['measured'] / s_R / 1e3
        L_Rs = p['static'] / s_R / 1e3
        t = p['t']
        # BOTH SOLID, same weight: the colour carries the distinction, so
        # styling them differently would say the comparison is not
        # like-for-like when it is.
        ax.plot(t, L_T, '-o', color=C_TRENCH, lw=2.0, ms=4.0,
                markeredgecolor='white', markeredgewidth=0.5,
                label='trench pull')
        ax.plot(t, L_Rd, '-o', color=C_RIDGE, lw=2.0, ms=4.0,
                markeredgecolor='white', markeredgewidth=0.5,
                label='ridge push (measured)')
        ax.set_title(f'({"ab"[row]})  {key}', fontsize=12, loc='left')
        ax.grid(alpha=0.25, color=C_RULE, lw=0.6)
        ax.set_ylabel(r'Effective arm,  $a_{\mathrm{eff}}$  [km]',
                      fontsize=12)
        ax.tick_params(labelsize=11)
        if row == 0:
            ax.legend(frameon=False, fontsize=11, ncol=2, loc='lower right')
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
        # KEYS RENAMED 2026-10-08, L -> a_eff. SYMBOLOGY reserves L for
        # domain length (§7.6), and the manuscript adopted a_eff as the
        # symbol for this quantity, defined there as |Delta GPE*| / Delta P
        # -- identical to what is computed here. The old L_* keys are cited
        # in budget.tex and must be updated with this rerender.
        for name, arr in (('a_eff_trench_km', L_T),
                          ('a_eff_ridge_measured_km', L_Rd),
                          ('a_eff_ridge_static_km', L_Rs),
                          ('relief_trench_m', s_T / RHO_G),
                          ('relief_ridge_m', s_R / RHO_G)):
            rows += [(key, f'{name}_median', f'{np.median(arr):.2f}'),
                     (key, f'{name}_q1', f'{np.percentile(arr, 25):.2f}'),
                     (key, f'{name}_q3', f'{np.percentile(arr, 75):.2f}')]
    axes[1].set_xlabel('Model time [Myr]', fontsize=12)
    # The axis names the quantity (SYMBOLOGY §2, 'effective arm'); the
    # manuscript defines it as a_eff = |Delta GPE*| / Delta P.
    # Minimal title naming the subject only; the interpretation ('how
    # efficiently topography converts...') and the domain parentheticals
    # were caption material and carried the retired domain wording.
    fig.suptitle('Effective arm', fontsize=13)
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
