"""fig_slab_buoyancy — slab buoyancy, potential-energy release, and where it happens.

MANUSCRIPT FIGURE (promoted from diag_ on 2026-09-24, Dan: "definite
include for the SI, maybe even main"). Writes figures/fig_slab_buoyancy.png
and tables/slab_buoyancy.csv from the slab cache
(scripts/kinematics/slab_geometry_cache.py -- the cache stays in the
kinematics directory; only the figure is manuscript material).

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
robust FORCE can be recovered from it by dividing by a representative
sinking rate. The reference is <v_z>_UM, the buoyancy-weighted rate over
the MID-UPPER MANTLE, 250-450 km -- unambiguously slab, centred on the
depth at which <v_z>(z) peaks (~330 km in both runs), and the SAME
quantity the velocity panel plots. The subscript marks it as a
representative average rather than a local value; an overbar is not used
because in this project's register the overbar denotes a depth integral.

⚠ THE REFERENCE IS ONE CONSTANT PER RUN, not a per-snapshot value. A
moving denominator would make F_eff(t) mix two signals -- dE/dt and
<v_z> rise together when the slab accelerates, so the ratio partly
cancels its own variation. Held fixed, F_eff is the release rate in force
units and its time dependence is the release rate's alone.

⚠ EACH RUN USES ITS OWN REFERENCE (1.67 STD / 1.99 WAL cm/yr), so F_eff
is a within-model participation measure. Do NOT read the STD-vs-WAL
difference in F_eff as a difference in buoyancy doing work: the release
rates are nearly equal (27.1 vs 27.6 kW/m) and almost all of the F_eff
gap is the reference difference. For a cross-model force comparison use
the release rates directly, or a shared reference.

    F_eff = (dE/dt) / (g * <v_z>_UM),  <v_z>_UM = the RUN MEDIAN of the
                                       buoyancy-weighted sinking rate over
                                       250-450 km -- one constant per run

It is the buoyancy that is actually doing work, expressed as a force. Its
spread across the same z_top range is -10 % / -9 % rather than -40 % /
-45 %: it discounts the translating plate automatically instead of
excluding it by hand.

⚠ It sits BELOW every value of F, at 51 (STD) / 44 (WAL) TN/m against
71-119 / 59-104, because the reference rate is taken where the slab sinks
FASTEST. F_eff is therefore a conservative measure of the buoyancy doing
work, not an estimate of the total present. Both are reported; the
argument needs only that one is convention-dependent and the other is
not.

  1  buoyancy force: the z_top band (wide) against the effective buoyancy
     (narrow). TWO QUANTITIES ONLY. An earlier version also carried the
     lower-mantle curve and the accumulated total; both were cut on
     2026-09-24 (Dan: "quite a lot going on there"). The lower mantle is
     already in row 2, and the accumulated totals have their own figure,
     kinematics/diag_buoyancy_time. The contrast between the two band
     WIDTHS is the whole content of this row and anything else competes
     with it.
  2  PE release rate, upper and lower mantle. THE UPPER-MANTLE CURVE
     CARRIES THE SAME z_top BAND as row 1 -- it is ~10 % wide against
     row 1's ~45 %, which is the comparison the figure exists to make, so
     both bands are labelled and edged.
  3  where the release happens: dE/dz against depth and time, 660 marked.

WHAT IT SHOWS.

  - **The Dahlen gap, quantified.** Upper-mantle buoyancy is 86 (STD) /
    72 (WAL) TN/m at z_top = 100 km, or 51 / 44 TN/m on the conservative
    effective measure -- against the 2-3 TN/m that reaches the trailing
    plate in the force balance. Between fifteen and forty times the
    delivered force is consumed within the subduction system, depending
    which measure and which run.
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

ACCUMULATED BUOYANCY -- AND WHY IT IS NOT A TIME-INTEGRATED FLUX
(Dan's question, 2026-09-24). The grey curve in row 1 is the total
buoyancy of cold material below z_top: a STOCK, and the right answer to
"which model has put more buoyancy into the mantle".

  total below 100 km    STD  35.1 -> 223.7 TN/m   (gain 188.6)
                        WAL  55.1 -> 244.5 TN/m   (gain 189.5)

⚠ It is tempting to obtain the same thing by integrating the buoyancy
flux across z_top over time. That is WRONG HERE and the arithmetic says
so: the advective input across 100 km totals 131 (STD) / 133 (WAL) TN/m
against stock gains of 189 / 190. The missing third is not an error --
cold material is also CREATED below z_top in place, by the plate's
thermal boundary layer thickening past that depth as it ages, over the
whole plate rather than at the trench. A flux across one horizon cannot
see a distributed source. Quote the stock.

⚠⚠ THE TWO MODELS ACCUMULATE AT THE SAME RATE. The gains over the run
are 188.6 and 189.5 TN/m -- indistinguishable. WAL's higher total is a
head start: its slab reached 660 km at ~1.5 Myr against ~4-5 Myr in STD
(Cerpa et al.), so it was already 20 TN/m ahead at t = 8. The weak layer
does not change how much buoyancy enters the mantle.

WHERE THEY DO DIFFER IS THE DEPTH IT GOES TO. dE/dz at a single depth
has units of buoyancy flux (kg/s^3 = N/m/s), so its time integral is a
force per unit length; across 660 km that gives the buoyancy each model
has delivered to the lower mantle:

  cumulative transfer across 660 km   STD  90.9    WAL  120.4 TN/m

WAL moves 32 % more through the transition zone on the same input. That
is the folding-regime contrast stated as a budget.

⚠ Delta rho > 0 only, so cold material that RISES contributes negative
release; this is correct and is why row 2 can dip.
⚠ The 5 km grid resolves the ~100 km anomaly with ~20 cells, adequate for
an integral but not for a gradient.
⚠ BOTH ROW-1 CURVES ARE MEASURES OF WHAT IS CONVENTIONALLY CALLED SLAB
PULL, and the axis now says so -- in quotes, because the paper's argument
is that this is NOT the force delivered to the trailing plate. Naming it
is what makes the contrast legible: a reader who does not recognise the
quantity cannot see that 86 TN/m of it yields 2-3 TN/m at the plate.
Never caption either curve as slab pull ON the trailing plate.

DRAFT CAPTION. Slab buoyancy and potential-energy release through the
runs, STD (left) and WAL (right). (a, b) Buoyancy force of cold material
in the upper mantle, shaded across upper integration depths $z_{top}$ of
75 to 125 km, against the effective buoyancy -- the release rate divided
by the slab's mean sinking rate over the mid-upper mantle,
$\langle v_z\rangle_{\mathrm{UM}}$ on 250-450 km -- shaded across the same
range. The wide band shows that the buoyancy force depends strongly on
where the plate is judged to end; the narrow one shows that the effective
buoyancy does not. (c, d) Rate of potential-energy release for the same
regions. Only the upper-mantle curve is shaded, because the lower-mantle
region is bounded at 660 km and below and so carries no dependence on
$z_{top}$; its shading spans the same range as (a, b) and is narrow
because the release rate is nearly independent of that choice. (e, f)
Release rate per unit depth against time, with the 660 km discontinuity
marked. Density anomalies are relative to ambient mantle at
3171.3 kg m^-3 and clipped at zero, so only cold material contributes.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.join(_HERE, 'kinematics'))
import slab_geometry_cache as sgc
from tables_io import write_table

ROOT = os.path.dirname(_HERE)
CM_S = 3.17098e-10           # 1 cm/yr in m/s
G = 9.8
Z_TOPS = (75e3, 100e3, 125e3)
Z_660 = 660e3
# ONE reference band for both the effective buoyancy and the velocity
# panel (Dan, 2026-09-24): the figure divides dE/dt by <v_z> and also plots
# <v_z>, so they must be the same quantity. Centred on the <v_z> maximum,
# which sits at ~330 km in both runs.
VZ_BAND = (250e3, 450e3)     # mid-upper mantle, the reference for <v_z>_UM
C_UM = '#0072B2'
C_LM = '#CC79A7'
C_EFF = 'k'
C_RULE = '#BFC3D1'
C_MODEL = {'STD': '#002147', 'WAL': '#E5007D'}   # FIGURE_STYLE encoding 2


def main():
    s = sgc.load()
    z = s['z_prof']; dz = z[1] - z[0]
    # SEQUENTIAL from zero, shared between the models. The release rate is
    # non-negative in both runs to rounding (STD min -3e-4, WAL exactly 0),
    # so an earlier diverging scale with a hardcoded negative vmin showed a
    # negative range that does not exist. The guard below fires if a rebuild
    # ever produces real negatives -- cold material rising -- at which point
    # a diverging scale becomes the honest choice again.
    global VMAX
    _all = np.concatenate([s[f'{k}_dEdz'].ravel() for k in ('STD', 'WAL')]) * 1e3
    VMAX = _all.max()
    if _all.min() < -0.01 * VMAX:
        print(f'  ** dE/dz has real negatives (min {_all.min():.3f}); '
              f'switch back to a diverging scale')
    # constrained layout, not tight_layout: the row-3 colourbar spans both
    # panels and tight_layout places such a colourbar inside the axes.
    # 3 rows, columns are models. A 4th row overlaying the models was tried
    # 2026-09-24 and removed the same day: it distracted from the three
    # panels that carry the figure. The depth-partition comparison it held
    # now has its own figure, kinematics/diag_buoyancy_time.
    fig, axes = plt.subplots(3, 2, figsize=(13.0, 11.6), sharex='col',
                             layout='constrained',
                             gridspec_kw={'height_ratios': [1.15, 1.0, 1.25]})
    rows = [('model', 'quantity', 'value')]

    for col, key in enumerate(('STD', 'WAL')):
        t = s[f'{key}_t']; F = s[f'{key}_dFdz']; E = s[f'{key}_dEdz']
        band = lambda a, lo, hi: a[:, (z >= lo) & (z < hi)].sum(1) * dz
        # ONE CONSTANT REFERENCE RATE PER RUN (Dan, 2026-09-24), the run
        # MEDIAN of <v_z>_UM -- not a per-snapshot value. With a moving
        # denominator, F_eff(t) mixes two signals: when the slab speeds up
        # both dE/dt and <v_z> rise and the ratio partly cancels its own
        # variation. A fixed yardstick makes F_eff the release rate
        # faithfully expressed in force units, which is the only thing the
        # conversion is for. The median is used rather than the mean
        # because WAL's 62-70 Myr episode skews the mean.
        ref = (z >= VZ_BAND[0]) & (z < VZ_BAND[1])
        vref_t = E[:, ref].sum(1) / F[:, ref].sum(1)           # m/s, per step
        vref = np.median(vref_t)                               # m/s, constant
        Fum = np.array([band(F, zt, Z_660) / 1e12 for zt in Z_TOPS])
        Pum = np.array([band(E, zt, Z_660) / 1e3 for zt in Z_TOPS])
        Feff = np.array([band(E, zt, Z_660) / vref / 1e12 for zt in Z_TOPS])
        Flm, Plm = band(F, Z_660, z[-1] + dz) / 1e12, band(E, Z_660, z[-1] + dz) / 1e3
        # ACCUMULATED buoyancy: the stock of cold material below z_top. This
        # is the quantity that answers "which model has put more buoyancy
        # into the mantle" -- see the docstring for why it is NOT the time
        # integral of the flux across z_top.
        Ftot = np.array([band(F, zt, z[-1] + dz) / 1e12 for zt in Z_TOPS])
        # Cumulative buoyancy TRANSFERRED across 660 km. dEdz at a single
        # depth has units of buoyancy flux (kg/s^3 = N/m/s), so its time
        # integral is a force per unit length.
        j660 = int(np.argmin(np.abs(z - Z_660)))
        f660 = E[:, j660]
        cum660 = np.concatenate([[0.0], np.cumsum(
            0.5 * (f660[1:] + f660[:-1]) * np.diff(t) * 3.15576e13)]) / 1e12

        a0, a1, a2 = (axes[r, col] for r in range(3))

        a0.fill_between(t, Fum[0], Fum[2], color=C_UM, alpha=0.28, lw=0,
                        label='upper mantle, $z_{top}$ = 75–125 km')
        for k_ in (0, 2):
            a0.plot(t, Fum[k_], '-', color=C_UM, lw=0.7, alpha=0.9)
        a0.plot(t, Fum[1], '-', color=C_UM, lw=2.0)
        a0.fill_between(t, Feff[0], Feff[2], color=C_EFF, alpha=0.30, lw=0,
                        label='effective, upper mantle: '
                              '$(dE/dt)/g\\langle v_z\\rangle_{\\mathrm{UM}}$')
        a0.plot(t, Feff[1], '-', color=C_EFF, lw=1.8)
        # Name the conventional term on the axis (Dan, 2026-09-24): both
        # curves in this row are measures of what the literature calls slab
        # pull, and the figure never said so. Quoted, because the paper's
        # argument is that it is NOT the force delivered to the plate.
        a0.set_ylabel('Buoyancy force —\n"slab pull" [TN/m]', fontsize=10)
        # PANEL LETTERS (Dan, 2026-09-28): the figure had none, so the text
        # could not direct the reader to a specific panel. Column-major:
        # (a,b,c) STD, (d,e,f) WAL.
        L = 'abc' if col == 0 else 'def'
        a0.set_title(f'({L[0]}) {key}', fontsize=11.5, loc='left')
        a1.set_title(f'({L[1]}) {key}', fontsize=11.5, loc='left')
        a2.set_title(f'({L[2]}) {key}', fontsize=11.5, loc='left')
        a0.legend(frameon=False, fontsize=8.5, loc='upper left')

        # The z_top band is drawn here too, and LABELLED AS SUCH. It is only
        # ~10 % wide, so without a legend entry and edge lines it reads as a
        # thick line -- which hides the very comparison the figure makes
        # (Dan, 2026-09-24). Row 1's band and this one span the same range.
        a1.fill_between(t, Pum[0], Pum[2], color=C_UM, alpha=0.28, lw=0,
                        label='upper mantle, $z_{top}$ = 75–125 km')
        for k_ in (0, 2):
            a1.plot(t, Pum[k_], '-', color=C_UM, lw=0.7, alpha=0.9)
        a1.plot(t, Pum[1], '-', color=C_UM, lw=2.2)
        # BOTH SOLID, distinguished by colour (Dan, 2026-09-24): a dashed
        # lower-mantle curve read as an arbitrary style difference. The one
        # real difference is that only the upper-mantle curve carries a
        # band, because the 660 km-to-base region has no z_top dependence
        # at all -- there is nothing to shade. The caption says so.
        a1.plot(t, Plm, '-', color=C_LM, lw=2.2, label='below 660 km')
        a1.axhline(0, color='k', lw=0.8)
        a1.set_ylabel('PE release rate [kW/m]', fontsize=10)
        a1.legend(frameon=False, fontsize=8.5, loc='upper left')

        # --- row 3: where the release happens -----------------------------
        # dEdz is stored as W/m per METRE of depth; x1000 for per km.
        # Colour scale is SHARED between the models so the panels compare
        # directly, and sequential from zero -- see main().
        Ei = E * 1e3
        im = a2.pcolormesh(t, z / 1e3, Ei.T, cmap='Reds', shading='nearest',
                           vmin=0.0, vmax=VMAX)
        a2.axhline(Z_660 / 1e3, color='k', lw=1.4, ls='--')
        a2.text(t[1], Z_660 / 1e3 - 25, '660 km', fontsize=8.5, va='bottom')
        a2.set_ylim(1800, 0)
        a2.set_ylabel('Depth [km]', fontsize=10)
        a2.set_xlabel('Model time [Myr]', fontsize=11)
        if col == 1:
            fig.colorbar(im, ax=axes[2, :].tolist(), orientation='horizontal',
                         pad=0.09, shrink=0.55, aspect=45,
                         label='$dE/dz$ [W m$^{-1}$ km$^{-1}$]')

        for ax in (a0, a1):
            ax.grid(alpha=0.25, color=C_RULE, lw=0.6)

        med = np.median
        print(f'== {key}  reference <v_z>_UM(250-450) = {vref/CM_S:.2f} cm/yr '
              f'(run median, constant)')
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
        print(f'   accumulated total below 100 km: {Ftot[1][0]:5.1f} -> '
              f'{Ftot[1][-1]:5.1f} TN/m (gain {Ftot[1][-1]-Ftot[1][0]:5.1f});  '
              f'cumulative transfer across 660 km {cum660[-1]:5.1f} TN/m')
        rows += [(key, 'lower_share_of_release_end',
                  f'{(Plm / (Plm + Pum[1]))[-1]:.3f}'),
                 (key, 'F_total_below100_start_TNm', f'{Ftot[1][0]:.2f}'),
                 (key, 'F_total_below100_end_TNm', f'{Ftot[1][-1]:.2f}'),
                 (key, 'F_total_below100_gain_TNm', f'{Ftot[1][-1]-Ftot[1][0]:.2f}'),
                 (key, 'cumulative_transfer_across_660_TNm', f'{cum660[-1]:.2f}'),
                 (key, 'F_lower_median_TNm', f'{med(Flm):.2f}'),
                 (key, 'dEdt_lower_median_kWm', f'{med(Plm):.2f}'),
                 (key, 'vz_ref_mid_upper_cmyr', f'{vref/CM_S:.3f}'),
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
    out = os.path.join(ROOT, 'figures', 'fig_slab_buoyancy.png')
    fig.savefig(out, bbox_inches='tight', dpi=200)
    print('written:', out)
    tab = write_table('slab_buoyancy', rows[0], rows[1:],
                      script='fig_slab_buoyancy.py',
                      figure='fig_slab_buoyancy.png',
                      models=('STD', 'WAL'),
                      meta={'rho_ambient': 3171.300, 'z_tops_km': [75, 100, 125],
                            'reference_band_km': [200, 660],
                            'identity': 'dE/dt = g F <v_z>',
                            'note': 'gross buoyancy, NOT the force delivered to '
                                    'the trailing plate'})
    print('written:', tab)


if __name__ == '__main__':
    main()
