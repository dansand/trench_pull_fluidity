"""diag_slab_buoyancy — slab buoyancy, potential-energy release, and where it happens.

DIAGNOSTIC (diag_, not yet a manuscript figure). Writes
figures/kinematics/diag_slab_buoyancy.png and tables/slab_buoyancy.csv
from the slab cache (scripts/kinematics/slab_geometry_cache.py).

Dan's request, 2026-09-24: quantify slab pull as a buoyancy force through
time, and the rate of potential-energy release with its depth dependence
-- particularly for WAL, to see whether the 62-70 Myr episode is a surge
in the upper or the lower mantle.

THE THREE QUANTITIES.

    F        = g * int(Delta rho) dA          buoyancy force      [TN/m]
    dE/dt    = g * int(Delta rho * v_z) dA    PE release rate     [kW/m]
    <v_z>    = (dE/dt) / (g * F)              buoyancy-weighted sinking rate

so that dE/dt = g * F * <v_z> EXACTLY. Delta rho is read from the model's
Density field (exactly linear in T; ambient 3171.300 kg/m^3) and clipped
at zero -- cold material only.

⚠ "SLAB PULL" IS NOT WELL POSED WITHOUT A DEPTH CONVENTION, AND THE PE
RELEASE RATE IS. Moving the top of the upper-mantle band from 75 to
125 km changes F by -40 % (STD) / -45 % (WAL) and dE/dt by only -10 % /
-9 %. The material in between is plate: it translates horizontally, so it
carries mass but does almost no work. Row 1 shows both bands on one axis
and the contrast between their widths is the point.

THE EFFECTIVE BUOYANCY (Dan, 2026-09-24). Because dE/dt is robust, a
robust FORCE can be recovered from it by dividing by a reference sinking
rate taken over a band that is unambiguously slab:

    F_eff = (dE/dt) / (g * <v_z>_{200-660 km})

It is the buoyancy that is actually doing work, expressed as a force. Its
spread across the same z_top range is -10 % / -9 % rather than -40 % /
-45 %, and it lands near the deep-z_top value of F -- it discounts the
translating plate automatically instead of excluding it by hand.

  1  buoyancy force: the z_top band (wide), the effective buoyancy (narrow),
     and the lower mantle.
  2  PE release rate, upper and lower mantle, with its own z_top band.
  3  where the release happens: dE/dz against depth and time, 660 marked.

WHAT IT SHOWS.

  - **The Dahlen gap, quantified.** Upper-mantle buoyancy is 86 (STD) /
    72 (WAL) TN/m at z_top = 100 km, or 72 / 60 TN/m as effective
    buoyancy -- against the 2-3 TN/m that reaches the trailing plate in
    the force balance. A factor of 25-35 is consumed within the
    subduction system.
  - **WAL puts twice as much into the lower mantle**: 62.8 against
    34.5 TN/m of buoyancy and 18.1 against 8.8 kW/m of release. That is
    the folding-into-the-lower-mantle regime of the source study showing
    up in the energy budget.
  - **The upper-mantle release is nearly identical between the runs**
    (27.6 vs 27.1 kW/m) despite WAL's plate moving 47 % faster. What the
    weak layer changes is the partition of release with depth, not the
    total available in the upper mantle.
  - **WAL's 62-70 Myr episode is a KINEMATIC surge, not a mass surge.**
    Upper-mantle dE/dt roughly doubles while the buoyancy force rises
    only a quarter: the same cold material falls faster. Row 3 localises
    it.

⚠ Delta rho > 0 only, so cold material that RISES contributes negative
release; this is correct and is why row 2 can dip.
⚠ The 5 km grid resolves the ~100 km anomaly with ~20 cells, adequate for
an integral but not for a gradient.
⚠ Gross buoyancy is NOT the force delivered to the plate. That is the
whole point of the contrast above, and the figure must never be captioned
as though it measured slab pull on the trailing plate.

DRAFT CAPTION. Slab buoyancy and potential-energy release through the
runs, STD (left) and WAL (right). (a, b) Buoyancy force of cold material
in the upper mantle, shaded across upper integration depths of 75 to
125 km, with the effective buoyancy -- the release rate divided by the
slab's mean sinking rate over 200-660 km -- shaded over the same range,
and the lower-mantle buoyancy below 660 km. The wide band shows that the
buoyancy force depends strongly on where the plate is judged to end; the
narrow one shows that the effective buoyancy does not. (c, d) Rate of
potential-energy release for the same regions. (e, f) Release rate per
unit depth against time, with the 660 km discontinuity marked.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE)); sys.path.insert(0, _HERE)
import slab_geometry_cache as sgc
from tables_io import write_table

ROOT = os.path.dirname(os.path.dirname(_HERE))
CM_S = 3.17098e-10           # 1 cm/yr in m/s
G = 9.8
Z_TOPS = (75e3, 100e3, 125e3)
Z_660 = 660e3
REF_BAND = (200e3, 660e3)    # unambiguously slab, for the reference v_z
C_UM = '#0072B2'
C_LM = '#CC79A7'
C_EFF = 'k'
C_RULE = '#BFC3D1'


def main():
    s = sgc.load()
    z = s['z_prof']; dz = z[1] - z[0]
    global NORM
    _lim = max(np.abs(s[f'{k}_dEdz']).max() for k in ('STD', 'WAL')) * 1e3
    NORM = TwoSlopeNorm(vcenter=0.0, vmin=-0.25 * _lim, vmax=_lim)
    # constrained layout, not tight_layout: the row-3 colourbar spans both
    # panels and tight_layout places such a colourbar inside the axes.
    fig, axes = plt.subplots(3, 2, figsize=(13.0, 11.6), sharex='col',
                             layout='constrained',
                             gridspec_kw={'height_ratios': [1.15, 1.0, 1.25]})
    rows = [('model', 'quantity', 'value')]

    for col, key in enumerate(('STD', 'WAL')):
        t = s[f'{key}_t']; F = s[f'{key}_dFdz']; E = s[f'{key}_dEdz']
        band = lambda a, lo, hi: a[:, (z >= lo) & (z < hi)].sum(1) * dz
        ref = (z >= REF_BAND[0]) & (z < REF_BAND[1])
        vref = E[:, ref].sum(1) / F[:, ref].sum(1)              # m/s
        Fum = np.array([band(F, zt, Z_660) / 1e12 for zt in Z_TOPS])
        Pum = np.array([band(E, zt, Z_660) / 1e3 for zt in Z_TOPS])
        Feff = np.array([band(E, zt, Z_660) / vref / 1e12 for zt in Z_TOPS])
        Flm, Plm = band(F, Z_660, z[-1] + dz) / 1e12, band(E, Z_660, z[-1] + dz) / 1e3

        a0, a1, a2 = (axes[r, col] for r in range(3))

        a0.fill_between(t, Fum[0], Fum[2], color=C_UM, alpha=0.28, lw=0,
                        label='upper mantle, $z_{top}$ = 75–125 km')
        a0.plot(t, Fum[1], '-', color=C_UM, lw=2.0)
        a0.fill_between(t, Feff[0], Feff[2], color=C_EFF, alpha=0.30, lw=0,
                        label='effective buoyancy $(dE/dt)/g\\langle v_z\\rangle$')
        a0.plot(t, Feff[1], '-', color=C_EFF, lw=1.8)
        a0.plot(t, Flm, '--', color=C_LM, lw=2.0, label='below 660 km')
        a0.set_ylabel('Buoyancy force [TN/m]', fontsize=10)
        a0.set_title(key, fontsize=11.5)
        a0.legend(frameon=False, fontsize=8.5, loc='upper left')

        a1.fill_between(t, Pum[0], Pum[2], color=C_UM, alpha=0.28, lw=0)
        a1.plot(t, Pum[1], '-', color=C_UM, lw=2.2, label='upper mantle')
        a1.plot(t, Plm, '--', color=C_LM, lw=2.0, label='below 660 km')
        a1.axhline(0, color='k', lw=0.8)
        a1.set_ylabel('PE release rate [kW/m]', fontsize=10)
        a1.legend(frameon=False, fontsize=8.5, loc='upper left')

        # --- row 3: where the release happens -----------------------------
        # dEdz is stored as W/m per METRE of depth; x1000 for per km.
        # Colour scale is SHARED between the models (computed over both in
        # main() below) so the panels can be compared directly.
        Ei = E * 1e3
        im = a2.pcolormesh(t, z / 1e3, Ei.T, cmap='RdBu_r', shading='nearest',
                           norm=NORM)
        a2.axhline(Z_660 / 1e3, color='k', lw=1.4, ls='--')
        a2.text(t[1], Z_660 / 1e3 - 25, '660 km', fontsize=8.5, va='bottom')
        a2.set_ylim(1800, 0)
        a2.set_ylabel('Depth [km]', fontsize=10)
        a2.set_xlabel('Model time [Myr]', fontsize=11)
        if col == 1:
            fig.colorbar(im, ax=axes[2, :].tolist(), pad=0.02, shrink=0.92,
                         label='$dE/dz$ [W m$^{-1}$ km$^{-1}$]')

        for ax in (a0, a1):
            ax.grid(alpha=0.25, color=C_RULE, lw=0.6)

        med = np.median
        print(f'== {key}  <v_z>(200-660) median {med(vref)/CM_S:.2f} cm/yr')
        for i, zt in enumerate(Z_TOPS):
            print(f'   z_top {zt/1e3:3.0f}: F {med(Fum[i]):6.1f}   '
                  f'F_eff {med(Feff[i]):6.1f} TN/m   dE/dt {med(Pum[i]):5.1f} kW/m')
            rows += [(key, f'F_upper_ztop{int(zt/1e3)}_median_TNm', f'{med(Fum[i]):.2f}'),
                     (key, f'F_eff_ztop{int(zt/1e3)}_median_TNm', f'{med(Feff[i]):.2f}'),
                     (key, f'dEdt_upper_ztop{int(zt/1e3)}_median_kWm', f'{med(Pum[i]):.2f}')]
        print(f'   below 660 : F {med(Flm):6.1f} TN/m   dE/dt {med(Plm):5.1f} kW/m')
        print(f'   z_top spread 75->125:  F {100*med(Fum[2]/Fum[0]-1):+.0f} %   '
              f'dE/dt {100*med(Pum[2]/Pum[0]-1):+.0f} %   '
              f'F_eff {100*med(Feff[2]/Feff[0]-1):+.0f} %')
        rows += [(key, 'F_lower_median_TNm', f'{med(Flm):.2f}'),
                 (key, 'dEdt_lower_median_kWm', f'{med(Plm):.2f}'),
                 (key, 'vz_ref_200_660_median_cmyr', f'{med(vref)/CM_S:.3f}'),
                 (key, 'ztop_spread_F_percent', f'{100*med(Fum[2]/Fum[0]-1):.1f}'),
                 (key, 'ztop_spread_dEdt_percent', f'{100*med(Pum[2]/Pum[0]-1):.1f}'),
                 (key, 'ztop_spread_Feff_percent', f'{100*med(Feff[2]/Feff[0]-1):.1f}'),
                 (key, 'lower_share_of_release_median',
                  f'{med(Plm / (Plm + Pum[1])):.3f}')]

    for row in (0, 1):
        lo = min(ax.get_ylim()[0] for ax in axes[row, :])
        hi = max(ax.get_ylim()[1] for ax in axes[row, :])
        for ax in axes[row, :]:
            ax.set_ylim(lo, hi)

    fig.suptitle('Slab buoyancy and potential-energy release', fontsize=10.5,
                 color='0.35')
    out = os.path.join(ROOT, 'figures', 'kinematics', 'diag_slab_buoyancy.png')
    fig.savefig(out, bbox_inches='tight', dpi=200)
    print('written:', out)
    tab = write_table('slab_buoyancy', rows[0], rows[1:],
                      script='kinematics/diag_slab_buoyancy.py',
                      figure='kinematics/diag_slab_buoyancy.png',
                      models=('STD', 'WAL'),
                      meta={'rho_ambient': 3171.300, 'z_tops_km': [75, 100, 125],
                            'reference_band_km': [200, 660],
                            'identity': 'dE/dt = g F <v_z>',
                            'note': 'gross buoyancy, NOT the force delivered to '
                                    'the trailing plate'})
    print('written:', tab)


if __name__ == '__main__':
    main()
