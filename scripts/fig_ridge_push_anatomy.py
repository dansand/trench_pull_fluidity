"""fig_ridge_push_anatomy — ridge push as three additive areas, and the depth
over which the plate actually moves.

Writes figures/fig_ridge_push_anatomy.png and tables/ridge_push_anatomy.csv
from the committed caches (run column_profiles_cache.py and
lab_kinematics_cache.py first).

MAIN-TEXT FIGURE for the ridge-push and plate-tilting section (Dan,
2026-09-27). It replaces an earlier three-column draft that showed the
same content as cumulative integrals and offered the reader FOUR candidate
ridge-push values -- none of which was the one the paper actually reports.
Both faults are fixed here: the force is shown as AREA, the way it is read
off a stress profile, and there are THREE quantities, chosen so they ADD:

    A   the ridge push as reported: the measured anomaly integrated to
        z_c = 75 km
    T   the tilt: the area between the measured and tilt-removed curves
        over the same depth, identically |Delta P| z_c
    S   the deep tail: what the tilt-removed curve still contributes
        between z_c and 200 km

    A + T + S  =  the tilt-removed anomaly integrated to 200 km, the
                  static limit.

Mid-run (36-44 Myr) values, TN/m:

              A       T       S     A+T+S    density route to 200 km
      STD    1.21    0.62    0.20    2.04           2.39
      WAL    1.36    0.38    0.18    1.92           2.24

The argument, left panels:

 1. There is a boundary layer holding a topographic pressure gradient,
    measured in sigma_zz alone -- area A.
 2. It decomposes trivially. The offset between the solid and dashed
    curves IS the tilt, and that the dashed curve coincides with the
    independent density-route prediction is what licenses calling the
    remainder the density structure rather than a residual.
 3. A static treatment would claim A + T + S, about 1.7x and 1.4x the
    reported value. Area S is what a static reading adds by integrating
    past the plate.

and, right panels:

 4. Why S is not admissible: by 125 km the column moves at ~0.6 of the
    plate's speed and by 200 km it has reversed. Area S accumulates
    through material progressively decoupling from the plate it is
    supposed to be driving.

⚠ z_c = 75 km IS NOMINAL, NOT DECLARED-BY-DEFINITION (Dan's wording,
2026-09-27). It is a single round depth standing in for the sigma_zz
maximum across both runs and the whole time span -- per snapshot that
maximum ranges over 32-126 km and trends upward with the boundary layer.
The SI figure fig_ridge_cumulative carries the justification and the
penalty: 75 km recovers 96-97 % of the per-snapshot optimum, median
shortfall 3 %.

⚠ THE VELOCITY PANEL IS NORMALISED by each snapshot's own plate speed.
Dimensionally the full-run band spans 1.1-2.9 (STD) and 1.6-4.8 (WAL)
cm/yr, which is the plate's speed history rather than the shape of the
profile, and it swamped the panel. Normalised, the band is shape variation
only. NO THRESHOLD IS DRAWN: an earlier draft marked a "dynamical LAB" at
a 10 % velocity departure, which is an ARBITRARY INTERNAL CRITERION
introduced in fig_lab_kinematics (LAB_FRAC, commit 758e02a) with no
external authority and no entry in ANALYSIS_CONVENTIONS. Grid lines let
the reader apply whatever fraction they prefer.

⚠ THE DENSITY ROUTE CARRIES NO ELEVATION MASS. The Fluidity mesh is
undeformed -- topography is a surface stress field -- so only the
deep-corrected shape of g int (rho_R - rho_I) dz is comparable, and its
deep constant is removed (cpc.derive's lith_R). Classically the absent
elevated wedge is worth g e (rho_m-rho_w) e/2 against a total
g e (rho_m-rho_w)(t/3 + e/2), i.e. 2 % here (Richter & McKenzie 1977,
eqs. 25-26). It is drawn as a thin check curve, NOT as a fourth value: it
differs from the tilt-removed curve by an rms of 2-3 MPa on a 45-48 MPa
signal, which integrates to ~0.3 TN/m by 200 km. That gap measures the
precision of the comparison, not a distinct force.

⚠ SHADING MEANS TWO DIFFERENT THINGS IN THIS FIGURE and the caption must
say so (paper agent, 2026-09-28). In the left and centre panels every fill
is a FORCE AREA; in the velocity panel the fill is the FULL-RUN RANGE.
There is no time band in the left or centre panels -- it was dropped when
the figure went to the three-area form, because it would have collided
with the areas. An earlier draft caption said "with the full analysed
interval shaded", which is true only of the velocity panel.

DRAFT CAPTION. The pressure anomaly and its force integral across the
isostatic domain, for STD (top) and WAL (bottom). All curves are averages
over the mid-run window, 36-44 Myr. Left: the vertical normal stress
anomaly of the ridge column
relative to the first isostatic column, as measured (solid) and with the
deep asthenospheric anomaly |Delta P| removed (dashed); the thin curve is
the independent prediction from the density structure alone. The shaded
areas are three alternative force estimates, NOT a time range: (1) the
ridge push as reported, the measured curve integrated to the fixed nominal
integration depth z_c = 75 km; (2) the plate-tilt reduction, the area
between the measured and de-tilted curves over the same depth, identically
|Delta P| z_c; and (3) the static ridge push, the de-tilted curve
integrated to 200 km. Centre: the corresponding cumulative integrals, with
the measured value marked at z_c and at its own maximum. Right: the
horizontal
velocity averaged from the first isostatic column to the ridge, normalised
by the plate's own speed at each snapshot. The column retains most of the
plate speed only to about 90 km and has reversed by 200 km, so area S
accumulates through material progressively decoupled from the plate.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import column_profiles_cache as cpc
import lab_kinematics_cache as lkc
from fig_budget_time import T_MIN_MYR
from tables_io import write_table

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C_RIDGE = '#D55E00'          # isostatic domain (schematic)
C_DENS = '#7B3294'           # the density route, a check curve only
C_RULE = '#BFC3D1'
Z_DEEP_KM = 200.0            # where the tilt-removed anomaly has asymptoted


def main():
    d = cpc.load()
    kin = lkc.load()
    sm = lambda a: gaussian_filter1d(a, 2)
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 9.0), sharey=True,
                             sharex='col', layout='constrained',
                             gridspec_kw={'width_ratios': [1.15, 1.0, 0.85]})
    rows = [('model', 'quantity', 'value')]
    letters = [['a', 'b', 'c'], ['d', 'e', 'f']]

    for r, key in enumerate(('STD', 'WAL')):
        c = cpc.derive(d, key)
        z, t = c['z'], c['t']
        zkm = z / 1e3
        run = t >= T_MIN_MYR
        mid = c['mid'] & run
        a0, acu, a1 = axes[r]

        meas = sm(c['p_R'][mid].mean(axis=0)) / 1e6          # MPa
        dP = c['dP'][mid].mean() / 1e6                       # negative
        unt = meas - dP                                      # tilt removed
        dens = sm(c['lith_R'][mid].mean(axis=0)) / 1e6

        def I(a, lo, hi):
            w = (zkm >= lo) & (zkm <= hi)
            return float(np.trapz(a[w], zkm[w] * 1e3) / 1e6)  # MPa*m -> TN/m

        A = I(meas, 0, cpc.ZC_KM)                  # the reported ridge push
        T = I(unt, 0, cpc.ZC_KM) - A               # = |dP| * z_c, identically
        S = I(unt, cpc.ZC_KM, Z_DEEP_KM)           # the deep tail

        shal = zkm <= cpc.ZC_KM
        deep = (zkm >= cpc.ZC_KM) & (zkm <= Z_DEEP_KM)   # kept for the table
        a0.fill_betweenx(zkm[shal], 0, meas[shal], color=C_RIDGE,
                         alpha=0.38, lw=0, zorder=1,
                         label=f'(1)  reported ridge push   {A:.2f} TN/m')
        a0.fill_betweenx(zkm[shal], meas[shal], unt[shal], color=C_RIDGE,
                         alpha=0.16, lw=0, zorder=1,
                         label=f'(2)  plate-tilt reduction   {T:.2f}')
        # (3) is the area under the SHIFTED (tilt-removed) sigma_zz curve,
        # integrated to the asymptote -- the static ridge push (Dan,
        # 2026-09-27, reverting an intermediate version that used the
        # density curve instead). Drawn FIRST so (1) and (2) sit on top.
        # The three are alternatives, not components: they are not summed.
        w200 = zkm <= Z_DEEP_KM
        Pstat = I(unt, 0, Z_DEEP_KM)
        a0.fill_betweenx(zkm[w200], 0, unt[w200], color='0.45', alpha=0.18,
                         lw=0, zorder=0,
                         label=f'(3)  static ridge push   {Pstat:.2f}')
        a0.plot(meas, zkm, '-', color=C_RIDGE, lw=2.0, label='measured')
        a0.plot(unt, zkm, '--', color=C_RIDGE, lw=1.6,
                label=f'tilt removed ($|\\Delta P|$ = {abs(dP):.1f} MPa)')
        a0.plot(dens, zkm, '-', color=C_DENS, lw=1.1, alpha=0.9,
                label='density structure (check)')
        # the plotted quantity is PRESSURE-POSITIVE, so it is Delta(-sigma_zz),
        # not Delta sigma_zz; and a pointwise anomaly is not a force -- only
        # its cumulative integral is Delta GPE* (paper agent, 2026-09-27)
        a0.set_xlabel(r'$\Delta(-\sigma_{zz})$ [MPa]', fontsize=11.5)
        a0.set_ylabel('Depth [km]', fontsize=12)
        # NO "1 + 2 + 3 = static" annotation (Dan, 2026-09-27): the three
        # areas are alternatives a reader might integrate, not components to
        # be summed, and labelling them A/T/S implied an accounting identity
        # that the argument does not need.

        # ---- the cumulative integrals, retained in the main text (Dan,
        # 2026-09-27) because the area panel alone does not show WHERE the
        # force stops accumulating. The band between the tilt-removed and
        # density curves is the area the lithostatic approximation claims
        # and the model does not carry.
        cum = lambda a: np.concatenate([[0.0], np.cumsum(
            0.5 * (a[1:] + a[:-1]) * np.diff(zkm) * 1e3)]) / 1e6
        Cm, Cu, Cd = cum(meas), cum(unt), cum(dens)
        acu.plot(Cm, zkm, '-', color=C_RIDGE, lw=2.0, label='measured')
        acu.plot(Cu, zkm, '--', color=C_RIDGE, lw=1.6, label='tilt removed')
        acu.plot(Cd, zkm, '-', color=C_DENS, lw=1.1, label='density structure')
        kmax = int(np.argmax(Cm[zkm <= Z_DEEP_KM]))
        acu.plot(Cm[kmax], zkm[kmax], 'o', ms=6.5, color=C_RIDGE,
                 markeredgecolor='white', markeredgewidth=1.0, zorder=5)
        acu.set_xlabel('cumulative force [TN/m]', fontsize=11.5)
        gap = float(Cd[np.argmin(np.abs(zkm - Z_DEEP_KM))]
                    - Cu[np.argmin(np.abs(zkm - Z_DEEP_KM))])
        i20 = np.argmin(np.abs(zkm - 20.0))
        izc = np.argmin(np.abs(zkm - cpc.ZC_KM))
        above = float((Cd - Cu)[izc]) / gap
        # HOW MUCH THE NOMINAL DEPTH COSTS (Dan, 2026-09-27): the measured
        # cumulative read at z_c against its own maximum. Both points are
        # marked so the reader sees the curve is flat between them.
        at_zc, at_max = float(Cm[izc]), float(Cm[kmax])
        acu.plot(at_zc, cpc.ZC_KM, 's', ms=6, color=C_RIDGE, mfc='white',
                 markeredgewidth=1.4, zorder=6)
        # LEFT of the axis and just below z_c, where the panel is empty --
        # the annotation is about the depth it sits next to (Dan, 2026-09-27)
        acu.text(0.09, cpc.ZC_KM + 9,
                 f'at $z_c$   {at_zc:.2f} TN/m\n'
                 f'at the maximum   {at_max:.2f}\n'
                 f'({100*abs(at_max-at_zc)/at_max:.1f} %; $z^*$ = {zkm[kmax]:.0f} km)',
                 transform=blended_transform_factory(acu.transAxes, acu.transData),
                 fontsize=9.0, ha='left', va='top')

        # ---- the velocity, normalised by each snapshot's own plate speed --
        zk2 = kin['z'] / 1e3
        km = ((kin[f'{key}_t'] >= lkc.MIDRUN_MYR[0])
              & (kin[f'{key}_t'] <= lkc.MIDRUN_MYR[1]))
        kr = kin[f'{key}_t'] >= T_MIN_MYR
        vall = kin[f'{key}_vx']
        vn = vall / vall[:, zk2 < 20].mean(axis=1)[:, None]
        vplate = float(vall[km][:, zk2 < 20].mean())
        a1.fill_betweenx(zk2, vn[kr].min(axis=0), vn[kr].max(axis=0),
                         color='0.55', alpha=0.28, lw=0, label='full run')
        vm = vn[km].mean(axis=0)
        a1.plot(vm, zk2, '-', color='k', lw=2.0, label='mid-run mean')
        a1.set_xlabel(r'$\langle v_x\rangle\,/\,v_{\mathrm{plate}}$'
                      '   ($x_I$ to the ridge)', fontsize=11.5)
        a1.set_xlim(-0.25, 1.1)

        for j, ax in enumerate(axes[r]):
            ax.axhline(cpc.ZC_KM, color='0.30', lw=1.2)
            ax.axvline(0, color='k', lw=0.9)
            ax.grid(alpha=0.25, color=C_RULE, lw=0.6)
            ax.set_title(f'({letters[r][j]}) {key}', fontsize=11.5, loc='left')
        # on the FIRST column (Dan, 2026-09-27), right-hand side just below
        # z_c, where every curve has dropped below ~10 MPa
        a0.text(0.97, cpc.ZC_KM + 9, f'$z_c$ = {cpc.ZC_KM:.0f} km (fixed nominal)',
                fontsize=9.0, color='0.25',
                transform=blended_transform_factory(a0.transAxes, a0.transData),
                ha='right', va='top')
        # the plate speed the profile is normalised by, so nothing is lost.
        # Placed at the TOP LEFT, next to the plate itself: the normalised
        # profile hugs 1.0 there, so the left of the panel is empty above
        # ~30 km while every other corner carries the band or the legend.
        a1.text(0.90, 8.0, f'$v_{{\\mathrm{{plate}}}}$ = {abs(vplate):.2f} cm/yr',
                transform=blended_transform_factory(a1.transAxes, a1.transData),
                fontsize=9.0, color='0.25', ha='right', va='top')
        if r == 0:
            # lines first, then the areas; column 2 repeats the same line
            # styles so it carries no legend of its own
            h, l = a0.get_legend_handles_labels()
            o = ([i for i, x in enumerate(l) if not x.startswith('(')]
                 + [i for i, x in enumerate(l) if x.startswith('(')])
            a0.legend([h[i] for i in o], [l[i] for i in o], frameon=False,
                      fontsize=9.0, loc='lower right')
            a1.legend(frameon=False, fontsize=9.0, loc='lower right')

        sel = (zkm > 5) & (zkm < 220)
        rms = float(np.sqrt(np.mean((unt[sel] - dens[sel]) ** 2)))
        dens200 = I(dens, 0, Z_DEEP_KM)
        v125 = float(np.interp(125.0, zk2, vm))
        print(f'{key}: A {A:.2f} | T {T:.2f} | S {S:.2f} | A+T+S {A+T+S:.2f} TN/m'
              f'  (static / reported {(A+T+S)/A:.2f});  density route {dens200:.2f}'
              f' (rms vs tilt-removed {rms:.1f} MPa)')
        print(f'      v/v_plate at 125 km = {v125:.2f};  |dP| = {abs(dP):.1f} MPa')
        print(f'      at z_c {at_zc:.2f} vs at the maximum {at_max:.2f} TN/m '
              f'({100*abs(at_max-at_zc)/at_max:.1f} %, z* = {zkm[kmax]:.0f} km); '
              f'static push {Pstat:.2f}')
        print(f'      lithostatic gap {gap:.2f} TN/m, {100*above:.0f} % above z_c; '
              f'surface: density demands {dens[0]:.1f} MPa, stress supplies {unt[0]:.1f} '
              f'({100*unt[0]/dens[0]:.0f} %)')
        rows += [(key, 'A_reported_ridge_push_TNm', f'{A:.3f}'),
                 (key, 'T_tilt_TNm', f'{T:.3f}'),
                 (key, 'S_deep_tail_TNm', f'{S:.3f}'),
                 (key, 'static_limit_A_T_S_TNm', f'{A+T+S:.3f}'),
                 (key, 'static_over_reported', f'{(A+T+S)/A:.3f}'),
                 (key, 'density_route_to_200km_TNm', f'{dens200:.3f}'),
                 (key, 'density_vs_untilted_rms_MPa', f'{rms:.2f}'),
                 (key, 'deep_dP_MPa', f'{dP:.2f}'),
                 (key, 'v_over_vplate_at_125km', f'{v125:.3f}'),
                 (key, 'zc_nominal_km', f'{cpc.ZC_KM:.0f}'),
                 (key, 'static_push_inside_density_curve_TNm', f'{Pstat:.3f}'),
                 (key, 'cumulative_at_zc_TNm', f'{at_zc:.3f}'),
                 (key, 'cumulative_at_maximum_TNm', f'{at_max:.3f}'),
                 (key, 'zc_vs_maximum_pct', f'{100*abs(at_max-at_zc)/at_max:.2f}'),
                 (key, 'zstar_midrun_km', f'{zkm[kmax]:.1f}'),
                 (key, 'lithostatic_gap_at_200km_TNm', f'{gap:.3f}'),
                 (key, 'lithostatic_gap_fraction_above_zc', f'{above:.3f}'),
                 (key, 'lithostatic_gap_0_20km_TNm', f'{float((Cd-Cu)[i20]):.3f}'),
                 (key, 'surface_density_demand_MPa', f'{dens[0]:.1f}'),
                 (key, 'surface_stress_supplied_MPa', f'{unt[0]:.1f}'),
                 (key, 'surface_supplied_over_demanded', f'{unt[0]/dens[0]:.3f}')]

    axes[0, 0].set_ylim(Z_DEEP_KM, 0)
    fig.suptitle('The isostatic-domain pressure anomaly and its force integral',
                 fontsize=11.5, color='0.35')
    out = os.path.join(ROOT, 'figures', 'fig_ridge_push_anatomy.png')
    fig.savefig(out, bbox_inches='tight', dpi=220)
    print('written:', out)
    tab = write_table('ridge_push_anatomy', rows[0], rows[1:],
                      script='fig_ridge_push_anatomy.py',
                      figure='fig_ridge_push_anatomy.png', models=('STD', 'WAL'),
                      meta={'curves': 'mid-run mean (36-44 Myr); shading is the full '
                                      'run, t >= 8 Myr',
                            'areas': 'A measured to z_c; T = |dP| z_c; S tilt-removed '
                                     'from z_c to 200 km; A+T+S is the static limit',
                            'zc': 'z_c = 75 km is NOMINAL, standing in for the sigma_zz '
                                  'maximum across both runs (see fig_ridge_cumulative)',
                            'density_route': 'a check curve, not a fourth value; it '
                                             'carries no elevation mass'})
    print('written:', tab)


if __name__ == '__main__':
    main()
