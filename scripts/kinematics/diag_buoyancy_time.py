"""diag_buoyancy_time — how much slab buoyancy each model puts into the mantle, and where.

DIAGNOSTIC (diag_). Writes figures/kinematics/diag_buoyancy_time.png and
tables/buoyancy_time.csv from the slab cache
(scripts/kinematics/slab_geometry_cache.py).

Dan's question, 2026-09-24: which model puts more buoyancy into the
mantle as a function of time? `diag_slab_buoyancy` answered it only as a
crossing of two curves in separate panels, which can be seen but not
read. This figure exists to make it a number.

THE MODELS ARE OVERLAID (FIGURE_STYLE encoding 2, STD navy / WAL
magenta), because every question here is a comparison between the runs.

  1  the accumulated buoyancy: total cold material below z_top, and its
     split into the upper mantle and below 660 km.
  2  the cumulative buoyancy transferred across 660 km, and the share of
     the release happening below it.

THE ANSWER, IN TWO PARTS.

  THEY ACCUMULATE AT THE SAME RATE. Total buoyancy below 100 km goes
  35.1 -> 223.7 TN/m in STD and 55.1 -> 244.5 in WAL: gains of 188.6 and
  189.5, indistinguishable. WAL's higher total is a HEAD START, not a
  higher rate -- its slab reached 660 km at ~1.5 Myr against ~4-5 Myr in
  STD (Cerpa et al.), so it was already ~20 TN/m ahead at t = 8. The weak
  asthenospheric layer does not change how much buoyancy enters.

  THEY DIFFER IN THE DEPTH IT REACHES. Cumulative transfer across 660 km
  is 90.9 (STD) against 120.4 (WAL) TN/m -- 32 % more on the same input --
  and the lower-mantle share of the release ends at 0.47 against 0.57.
  That is the folding regime of the source study stated as a budget.

⚠ THE STOCK IS NOT A TIME-INTEGRATED FLUX, and the arithmetic says so:
advective input across 100 km totals 131 (STD) / 133 (WAL) TN/m against
stock gains of 189 / 190. Cold material is also CREATED below z_top in
place, as the plate's thermal boundary layer thickens past that depth
along its whole length. A flux across one horizon cannot see a
distributed source. Panel 1 is a stock; panel 2 is a flux integral across
a horizon the plate does not straddle, where the same objection does not
apply.

⚠ The record starts at t = 8 Myr, so panel 2 is cumulative FROM 8 Myr and
both curves start at zero by construction. Panel 1's t = 8 values carry
what accumulated before that and are the reason the totals differ.

⚠ dE/dz at a single depth has units of buoyancy flux (kg s^-3 = N m^-1
s^-1); its time integral is the force per unit length plotted in panel 2.
Same array, two readings -- see the cache docstring.

⚠ Gross buoyancy is NOT the force delivered to the trailing plate: 2-3
TN/m of the ~200 TN/m here reaches the plate in the force balance.

DRAFT CAPTION. Slab buoyancy through the runs, STD (navy) and WAL
(magenta). (a) Accumulated buoyancy of cold material below 100 km depth
(solid), split into the upper mantle above 660 km (thin) and the lower
mantle below it (dashed). The two runs accumulate at the same rate -- the
gains over the record are 189 and 190 TN/m -- and WAL's higher total is a
head start from its earlier arrival at the transition zone. (b) Cumulative
buoyancy transferred across 660 km (solid, left axis) and the share of
potential-energy release occurring below that depth (dotted, right axis).
The weak asthenospheric layer does not change how much buoyancy enters the
mantle; it changes how deep it gets.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE)); sys.path.insert(0, _HERE)
import slab_geometry_cache as sgc
from tables_io import write_table

ROOT = os.path.dirname(os.path.dirname(_HERE))
Z_TOP = 100e3
Z_660 = 660e3
SEC_PER_MYR = 3.15576e13
C = {'STD': '#002147', 'WAL': '#E5007D'}        # FIGURE_STYLE encoding 2
C_RULE = '#BFC3D1'


def main():
    s = sgc.load()
    z = s['z_prof']; dz = z[1] - z[0]
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 8.2), sharex=True,
                             layout='constrained',
                             gridspec_kw={'height_ratios': [1.35, 1.0]})
    a0, a1 = axes
    a1b = a1.twinx()
    rows = [('model', 'quantity', 'value')]

    for key in ('STD', 'WAL'):
        t = s[f'{key}_t']; F = s[f'{key}_dFdz']; E = s[f'{key}_dEdz']
        band = lambda a, lo, hi: a[:, (z >= lo) & (z < hi)].sum(1) * dz
        Ftot = band(F, Z_TOP, z[-1] + dz) / 1e12
        Fum = band(F, Z_TOP, Z_660) / 1e12
        Flm = band(F, Z_660, z[-1] + dz) / 1e12
        Pum = band(E, Z_TOP, Z_660) / 1e3
        Plm = band(E, Z_660, z[-1] + dz) / 1e3
        j660 = int(np.argmin(np.abs(z - Z_660)))
        f660 = E[:, j660]
        cum = np.concatenate([[0.0], np.cumsum(
            0.5 * (f660[1:] + f660[:-1]) * np.diff(t) * SEC_PER_MYR)]) / 1e12
        share = Plm / (Plm + Pum)

        a0.plot(t, Ftot, '-', color=C[key], lw=2.8, label=f'{key}  total')
        a0.plot(t, Fum, '-', color=C[key], lw=1.2, alpha=0.85,
                label=f'{key}  upper mantle')
        a0.plot(t, Flm, '--', color=C[key], lw=1.6,
                label=f'{key}  below 660 km')
        a0.annotate(f'{Ftot[-1]:.0f}', xy=(t[-1], Ftot[-1]),
                    xytext=(6, 0), textcoords='offset points', va='center',
                    color=C[key], fontsize=10.5, fontweight='bold',
                    annotation_clip=False)

        a1.plot(t, cum, '-', color=C[key], lw=2.6, label=key)
        a1.annotate(f'{cum[-1]:.0f}', xy=(t[-1], cum[-1]),
                    xytext=(6, 0), textcoords='offset points', va='center',
                    color=C[key], fontsize=10.5, fontweight='bold',
                    annotation_clip=False)
        a1b.plot(t, share, ':', color=C[key], lw=1.8)

        gain = Ftot[-1] - Ftot[0]
        print(f'{key}: total below {Z_TOP/1e3:.0f} km  {Ftot[0]:5.1f} -> '
              f'{Ftot[-1]:5.1f} TN/m (gain {gain:5.1f});  cumulative across '
              f'660 km {cum[-1]:5.1f} TN/m;  lower-mantle share of release '
              f'{share[-1]:.2f} at the end, {np.median(share):.2f} median')
        rows += [(key, 'F_total_start_TNm', f'{Ftot[0]:.2f}'),
                 (key, 'F_total_end_TNm', f'{Ftot[-1]:.2f}'),
                 (key, 'F_total_gain_TNm', f'{gain:.2f}'),
                 (key, 'F_upper_end_TNm', f'{Fum[-1]:.2f}'),
                 (key, 'F_lower_end_TNm', f'{Flm[-1]:.2f}'),
                 (key, 'cumulative_across_660_TNm', f'{cum[-1]:.2f}'),
                 (key, 'lower_share_of_release_end', f'{share[-1]:.3f}'),
                 (key, 'lower_share_of_release_median', f'{np.median(share):.3f}')]

    a0.set_ylabel(f'Accumulated buoyancy below {Z_TOP/1e3:.0f} km [TN/m]',
                  fontsize=10)
    a0.legend(frameon=False, fontsize=8.5, loc='upper left', ncol=2)
    a0.grid(alpha=0.25, color=C_RULE, lw=0.6)

    a1.set_ylabel('Cumulative buoyancy\nacross 660 km [TN/m]', fontsize=10)
    a1.set_xlabel('Model time [Myr]', fontsize=11)
    a1.legend(frameon=False, fontsize=9.5, loc='upper left')
    a1.grid(alpha=0.25, color=C_RULE, lw=0.6)
    a1b.set_ylabel('share of release below 660 km\n(dotted)', fontsize=9)
    a1b.set_ylim(0, 1)

    fig.suptitle('Slab buoyancy through time', fontsize=10.5, color='0.35')
    out = os.path.join(ROOT, 'figures', 'kinematics', 'diag_buoyancy_time.png')
    fig.savefig(out, bbox_inches='tight', dpi=220)
    print('written:', out)
    tab = write_table('buoyancy_time', rows[0], rows[1:],
                      script='kinematics/diag_buoyancy_time.py',
                      figure='kinematics/diag_buoyancy_time.png',
                      models=('STD', 'WAL'),
                      meta={'z_top_km': Z_TOP / 1e3, 'rho_ambient': 3171.300,
                            'panel_a': 'stock of cold material (NOT a time-'
                                       'integrated flux; see docstring)',
                            'panel_b': 'time integral of the buoyancy flux '
                                       'across 660 km, from t = 8 Myr'})
    print('written:', tab)


if __name__ == '__main__':
    main()
