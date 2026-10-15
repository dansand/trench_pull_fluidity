"""fig_balance_snapshot_absolute — the trailing-plate force balance at the
reference snapshot (t = 40 Myr, conventions §4b mid-run rule).

Writes figures/fig_balance_snapshot.png. Reads the archive directly
(one snapshot per model); computation lifted from
notebooks/fluidity_single_step.ipynb §§7–8 (single implementation of the
resultants; the notebook remains freely re-runnable at any step).

Rendering follows the NOTEBOOK's developed figures (Dan's directive
2026-09-15: the notebook images are the starting point, never
reinvented): panel styling, colours, labels, and axis limits lifted from
fluidity_single_step.ipynb cells §8.1b (fundamental form: F_B red,
Δσ̄_xx blue, sum green dashed) and §8.2 (decomposed form: ΔN_D black,
ΔGPE* blue thick, F_B red, residual green dashed), topography panel with
the x_T (navy) / x_I / x_R annotations. SEAWARD SIDE ONLY — the landward
side is distracting (Dan).

THE FREE SURFACE IS READ AT THE TOP-BOUNDARY NODES (Dan, 2026-09-21),
via fluidity_helpers.surface_fs, NOT as the top row of the sampled grid.
That row sits at z = 500 m and this model has no deforming mesh, so it
was reading the interior extension of a boundary field, not the surface.
Effect at the reference snapshot: w_T 629 -> 1761 m (STD) and
1062 -> 1453 m (WAL); x_I 83 -> 93 km and 100 -> 97 km; and the spurious
~11 km wiggle through the trench zone disappears, so the display
smoothing that existed to tame it is gone too. Corroborated two ways:
the corrected depths sit in the 1.6-2.3 km dry-trench range the project
quotes, and the independent equivalent topography sigma_zz(0)/rho_g
agrees with this reading to 1 m across the whole plate.

⚠ The column cache (column_profiles_cache.py) and fig_headline_tracking
STILL USE THE OLD READING, so their x_I differs from this figure's by a
few km until they are rebuilt.

SHEAR-SUPPORTED TOPOGRAPHY OVERLAY (Dan, 2026-09-21). The topography
panel carries w_tau = (dV/dx)/rho_m g (SYMBOLOGY §2), the deflection
required to balance the vertical shear load, lifted from
fluidity_single_step.ipynb's w_actual_vs_w_tau figure: V smoothed 10 km
before differentiating. BOTH curves are zero-referenced at x_I (Dan,
2026-09-21) — the column where dV/dx = 0, so the natural zero for a
deflection and for its shear-supported counterpart alike, and the
notebooks' own convention (w = fs_top[iI] - fs_top). The sign is pinned
on the data (conventions §1.2),
because the repo's V is the negative of the register's (SYMBOLOGY §7.5)
and the formula's sign therefore depends on which V is in hand.

It accounts for 90 % of the measured trench deflection in BOTH runs
(1733 of 1919 m STD, 1424 of 1582 m WAL), r = +1.00 over
s = 0..400 km seaward of the trench: the trench is a
shear-supported load, and dynamic topography is not needed to explain
it. The two curves separate toward the ridge, where the relief is
isostatic cooling topography rather than shear-supported.

NAMING: Dan called this the "equivalent topography" on 2026-09-21, but
W21 already assigned that term to the topography implied by the SURFACE
NORMAL STRESS, sigma_zz(0)/rho_g — a different quantity (which is what
validated the surface reading above). This figure therefore uses the
symbol SYMBOLOGY §2 already defines, w_tau, and calls it the
shear-supported topography; the collision is flagged for a ruling.

The overlay is OPTIONAL: set SHOW_W_TAU = False to drop it (and its
legend) and leave the topography panel as the surface alone.

SQUARE-ROOT x AXIS (Dan, 2026-09-21; xlim 0..3500 km). x_I sits 83 km
(STD) / 100 km (WAL) from the trench while the ridge is at ~2900 km, so
on a linear axis the NON-ISOSTATIC DOMAIN — where the whole trench pull
is generated — occupied under 3 % of the plot width and could not be
read. Alternatives considered and rejected: a broken axis (honest, but
three stacked rows would each need the break, and it cuts the curves);
symlog (expands most, but distorts hardest and needs an arbitrary
linthresh). sqrt(x) is continuous and monotone, so the curves stay
single and unbroken, and it moves x_I to ~18 % of the width.

⚠ GRADIENTS ARE NOT COMPARABLE ACROSS THIS AXIS. F_B accumulates very
nearly linearly in x, but on a sqrt axis it appears to flatten seaward;
the apparent steepening of every curve near the trench is likewise part
scale, part signal. Read VALUES and CROSSINGS off this figure, never
slopes. The axis label names the scale for that reason. Tick positions
are round numbers chosen to be near-evenly spaced once square-rooted,
otherwise the near-trench labels collide.

Decomposed panel in the PURE Δ FORM (Dan's simplification 2026-09-15):
every term zero at the trench — no renormalised/absolute-anchored
variant in this figure; the trench VALUES are communicated separately by
fig_trench_resultants. Because ΔN_D and ΔGPE* nearly coincide (F_B is
small), each panel carries a force-direction glyph.

THE GLYPH IS SNAPSHOT-SPECIFIC (Dan, 2026-09-21). It was a sign key —
"a positive value of this term would mean a force this way". It is now a
statement about the model state at the plotted step: each arrow is drawn
in the direction that term ACTUALLY acts, computed from its plate-wide
value at the ridge column, per model. Change STEP and the arrows follow.
In the mirrored analysis frame, leftward = trench-ward = driving:
  ΔGPE*  positive -> LEFT.  Positive at every step of both runs, so in
         practice always leftward — but computed, never hard-coded.
  ΔN_D   positive -> RIGHT. The ONLY genuinely variable arrow; it flips
         leftward whenever the plate-wide ΔN_D goes negative. Positive
         in both runs at the t = 40 Myr reference (+1.20 STD / +2.08 WAL).
  F_B    positive -> RIGHT. Always positive here (+2.20 / +1.15).
  Δσ̄_xx  positive -> RIGHT. Negative at the reference (−2.21 / −1.10),
         so it draws LEFTWARD — the fundamental form's single driving
         term, balancing F_B.
BOTH force panels carry a glyph (Dan, 2026-09-21): the middle one in the
fundamental form (Δσ̄_xx against F_B), the lower one in the decomposed
form. Every direction is printed at build time alongside its value, so a
flip is visible in the log and cannot pass silently into a figure.

Glyph placement: the header clearance and row spacing are set in INCHES
and converted to each axes' own fraction, because the panels differ in
height (ratios 1 : 1.4 : 1.8) while the text is a fixed point size — a
hard-coded fraction that clears the header in the tall panel runs
through it in the short one.

The displayed residual is the conventions §2.3 PINNED closure
(constant removed over x_T + 1000..2000 km, printed) — the near-trench
anchor noise is shown, not hidden.

Layout: columns STD | WAL; rows: topography · fundamental form ·
decomposed (Δ) form.

All Δ curves are trench-referenced (±5 km window means, conventions
§2.1/§4.1). Sign pin: ΔGPE* at x_I must reproduce the committed
time-evolution trench pulls at t = 40 (1.98 STD / 1.75 WAL TN/m) — asserted.

DRAFT CAPTION. The trailing-plate force balance at the reference
snapshot (t = 40 Myr) for STD (left) and WAL (right). Distance from the
trench is plotted on a square-root scale, which expands the non-isostatic
domain between the trench and the first isostatic column; gradients are
therefore not comparable along the axis. (a, b) Surface topography w, read
at the model's top boundary and positive downward (the axis is inverted,
so the trench is at the bottom), with the trench, first isostatic and
ridge columns marked. The first isostatic column x_I is the first column
seaward of the trench at which the free surface has recovered to its
far-field elevation; the zero of dV/dx is carried as a cross-check and
agrees to about 2 km. (c, d) The fundamental form of the vertically integrated
balance -- the change in the vertically integrated horizontal normal
stress against the accumulated basal traction. (e, f) The decomposed
form, Delta N_D - Delta GPE* + F_B = 0, every term referenced to zero at
the trench: the topographic pressure term carries the balance and
Delta N_D is secondary. The residual (green dashed), with its
trench-anchor constant removed over the declared mid-subducting-plate
window, shows where extraction is imperfect, principally the trench zone.
(g, h) The normal-stress-difference resultant as its own columnwise value
N_D(x), rather than as a difference -- the only row whose ordinate is a
single column's resultant rather than a force on the plate segment
between the trench and x. The panel is labelled tension-like
above zero and compression-like below: N_D is a difference between normal
stresses, so a positive value means the horizontal normal stress exceeds
the vertical one rather than that the plate is in tension. This shows what the Delta forms
cannot: the resultant is COMPRESSIONAL at the trench (-0.74 and
-1.78 TN/m in the two runs) and stays modest across the whole plate, so
the plate-wide change in N_D is not delivered as a large tension-like
resultant at the hinge. The arrows in (c-f) give the direction in which each term acts at
this time step.

VARIANT of fig_balance_snapshot (Dan, 2026-09-25): a fourth row added,
and the w_tau overlay turned OFF in the topography panel, which is
shortened accordingly. The parent figure keeps both. An older SI figure
named fig_balance_absolute is a separate rendering of the same
calculation -- check for overlap before placing both.
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tables_io import write_table

# THE CHAIN LIVES IN balance_chain (hoisted 2026-10-09). This script and
# fig_balance_snapshot_absolute.py each carried a byte-identical copy of
# load_snapshot and compute, and their glyph headers had drifted apart.
from balance_chain import (compute, draw_direction_glyph, ROOT, DX, ZC, W,
                           T_REF_MYR, SQRT_TICKS_KM, PIN_KM, RHO_M, G,
                           SHEAR_SMOOTH_KM, COMMITTED_TP_T40)
SHOW_W_TAU = False                    # OFF in this variant (Dan, 2026-09-25)

def main():
    if not hasattr(np, 'trapz'):
        np.trapz = np.trapezoid
    # The parent fig_balance_snapshot writes no table; this variant does,
    # because its caption quotes row-4 values (Dan's verifiable-numbers
    # pattern). Flagged: the parent's numbers are still unbacked.
    rows = [('model', 'quantity', 'value')]
    # FOUR rows (Dan, 2026-09-25); the topography row is shortened since
    # the w_tau overlay is off in this variant and it carries one curve.
    fig, axes = plt.subplots(4, 2, figsize=(10.5, 10.2), sharex=True,
                             gridspec_kw={'height_ratios': [0.72, 1.3, 1.6, 0.92]})
    lims = {0: [], 1: [], 2: [], 3: []}   # per-row plotted data, for axis limits
    for col, key in enumerate(('STD', 'WAL')):
        r = compute(key)
        x, xT, t_myr = r['x'], r['xT'], r['t_myr']
        fs_top, d_Sxx, d_gpe, d_Fd, FB = (r['fs_top'], r['d_Sxx'], r['d_gpe'],
                                          r['d_Fd'], r['FB'])
        res_pin = r['res_pin']
        iI_x, xR = r['xI'], r['xR']
        # --- render: lifted from the notebook cells (§8.1b, §8.2) ---
        # distance from the trench, POSITIVE SEAWARD. Native frame
        # (W31): seaward is -x, so this is x_T - x, not x - x_T.
        xkm = (xT - x) / 1e3
        xi_km, xr_km = (xT - iI_x) / 1e3, (xT - xR) / 1e3
        vis = (xkm >= -50) & (xkm <= 3500)      # the displayed span

        # topography: display-smoothed SEAWARD OF x_I ONLY — the
        # trailing-plate short-wavelength content otherwise dominates
        # visually (Dan), but the narrow trench is REAL structure and
        # filtering shaves hundreds of metres off w_T (the aspect lesson):
        # raw through the trench zone, blended into the smoothed curve
        # over 100 km beyond x_I.
        # No display smoothing any more (2026-09-21): the top-boundary
        # reading is already smooth, so the blend that existed to tame the
        # z = 500 m row's spurious ~11 km wiggle has nothing to tame. A
        # 9 km median moves the trench value by 2 m.
        topo_disp = r['w']          # zero at x_I

        ax1 = axes[0, col]
        if SHOW_W_TAU:
            ax1.plot(xkm, r['w_tau'], color='#E5007D', lw=1.3, ls='--',
                     label=r'$w_\tau = (dV/dx)/\rho_m g$')
        ax1.plot(xkm, topo_disp, color='k', lw=1.5, label='$w$')
        ax1.axhline(0, color='k', lw=0.5)
        for xc in (0, xi_km, xr_km):
            ax1.axvline(xc, color='k', lw=0.5)
        lims[0] += ([topo_disp[vis], r['w_tau'][vis]] if SHOW_W_TAU
                    else [topo_disp[vis]])
        ax1.set_title(f'{key},  $t = {t_myr:.1f}$ Myr',
                      fontsize=12)
        # THE x_I NOTE IS IN THE CAPTION, not on the panel (Dan,
        # 2026-10-15). It read 'first isostatic column (x_I), dV/dx = 0',
        # which named the cross-check rather than the extraction
        # (conventions §3.2 — free-surface recovery), and the column is
        # already marked by its rule and its x_I tick above it.
        # x_T now sits ON the left spine (the sqrt axis starts at 0), so its
        # label must go inside the axes, not to the left of the line
        for x_col, lab, c, side in [(0, r'$x_T$', '#002147', 'left'),
                                    (xi_km, r'$x_I$', 'k', 'left'),
                                    (xr_km, r'$x_R$', 'k', 'left')]:
            ax1.annotate(lab, xy=(x_col, 1), xycoords=('data', 'axes fraction'),
                         xytext=(5 if side == 'left' else -5, -4),
                         textcoords='offset points',
                         ha=side, va='top', color=c, fontsize=13,
                         fontweight='bold')

        ax2 = axes[1, col]
        ax2.plot(xkm, FB * 1e-12, color='red', lw=2, label=r'$F_B(x)$')
        ax2.plot(xkm, d_Sxx * 1e-12, color='b', lw=2, label=r'$\Delta\bar\sigma_{xx}(x)$')
        ax2.plot(xkm, (d_Sxx + FB) * 1e-12, color='g', ls='--', lw=3,
                 label=r'$\Delta\bar\sigma_{xx}(x) + F_B(x)$')
        ax2.axhline(0, color='k', lw=0.5)
        for xc in (0, xi_km, xr_km):
            ax2.axvline(xc, color='k', lw=0.5)
        lims[1] += [FB[vis] * 1e-12, d_Sxx[vis] * 1e-12, (d_Sxx + FB)[vis] * 1e-12]

        # Δ form (Dan, 2026-09-15): everything zero at the trench — the pure
        # communication of the balance; trench VALUES live in
        # fig_nd_trench_ridge, not here. ΔN_D and ΔGPE* plot nearly on top
        # of each other (F_B is small): the direction box carries the sign
        # reading so the coincidence is not misread as one curve.
        ax3 = axes[2, col]
        ax3.plot(xkm, res_pin * 1e-12, color='g', ls='--', lw=3,
                 label=r'$\Delta N_D - \Delta\mathrm{GPE}^{*} + F_B$')
        ax3.plot(xkm, FB * 1e-12, color='red', lw=2, label=r'$F_B(x)$')
        ax3.plot(xkm, d_gpe * 1e-12, color='b', lw=4, alpha=0.6,
                 label=r'$\Delta\mathrm{GPE}^{*}(x)$')
        ax3.plot(xkm, d_Fd * 1e-12, color='k', ls='-', lw=1.5,
                 label=r'$\Delta N_D(x)$')
        ax3.axhline(0, color='k', lw=0.5)
        for xc in (0, xi_km, xr_km):
            ax3.axvline(xc, color='k', lw=0.5)
        lims[2] += [res_pin[vis] * 1e-12, FB[vis] * 1e-12,
                    d_gpe[vis] * 1e-12, d_Fd[vis] * 1e-12]
        ax3.set_xlim(0, 3500)                   # SEAWARD ONLY (Dan, 2026-09-15)

        # ---- ROW 4: the ABSOLUTE columnwise resultant (Dan, 2026-09-25) --
        # Rows 2-3 are Delta forms, everything zero at the trench. Here N_D
        # is plotted as it stands and the GPE-like term is shifted onto it
        # by the trench value, so BOTH curves start at N_D(x_T) at x = 0.
        # N_D = N_D(x_T) + dGPE* - F_B, so THE GAP BETWEEN THEM IS F_B,
        # read off the panel rather than inferred from a third line.
        Fd, Fd_T = r['Fd'], r['Fd_T']
        gpe_shift = Fd_T + d_gpe
        # 0.3 TN/m: the residual peaks near the trench, the worst-resolved
        # column (conventions §2.3), at ~4 % of the terms. A wiring check,
        # not a precision claim.
        dev = float(np.abs(Fd - (gpe_shift - FB))[vis].max())
        assert dev < 3e11, (f'{key}: N_D != N_D(x_T) + dGPE* - F_B, '
                            f'max deviation {dev*1e-12:.3f} TN/m')
        # JUST N_D (Dan, 2026-09-25). The shifted GPE curve and the F_B
        # gap were dropped: row 3 already shows that the Delta forms track
        # each other, so repeating the comparison here added a second
        # reading of the same fact. What this row is FOR is the absolute
        # level -- where N_D actually sits -- and one curve says it.
        ax4 = axes[3, col]
        ax4.plot(xkm, Fd * 1e-12, color='k', lw=1.8, label=r'$N_D(x)$')
        ax4.axhline(0, color='k', lw=0.5)
        ax4.plot(0, Fd_T * 1e-12, 'o', color='k', ms=5, zorder=6)
        for xc in (0, xi_km, xr_km):
            ax4.axvline(xc, color='k', lw=0.5)
        lims[3] += [Fd[vis] * 1e-12]
        ax4.set_xlabel('Distance from trench [km]  ' + r'($\sqrt{x}$ scale)',
                       fontsize=12)
        ax4.set_xlim(0, 3500)
        print(f'   {key} row 4: N_D(x_T) {Fd_T*1e-12:+.2f} TN/m; N_D at x_R '
              f'{float(np.interp(xR, x, Fd))*1e-12:+.2f}; gap at x_R (= F_B) '
              f'{float(np.interp(xR, x, FB))*1e-12:+.2f}; identity holds to '
              f'{dev*1e-12:.3f} TN/m (the closure residual)')
        # THE SHEAR-SUPPORTED TOPOGRAPHY TEST (SI rows, manuscript audit
        # 2026-10-12). w_tau = (dV/dx)/(rho_m g) is the deflection a static
        # balance against the vertical shear load would require; comparing it
        # with the measured free surface at the trench is what licenses
        # calling the trench topography stress-supported. Both curves are
        # zero-referenced at x_I (balance_chain), and w_tau's sign is pinned
        # on the data there, not asserted. These numbers were printed and
        # discarded; the SI quotes them, so they are rows now.
        _ti = int(np.argmin(np.abs(r['x'] - r['xT'])))
        _wm, _wt = float(r['w'][_ti]), float(r['w_tau'][_ti])
        rows += [(key, 'w_trench_measured_m', f'{_wm:.0f}'),
                 (key, 'w_trench_shear_supported_m', f'{_wt:.0f}'),
                 (key, 'w_trench_shear_supported_fraction', f'{_wt/_wm:.3f}'),
                 (key, 'nd_trench_TNm', f'{Fd_T*1e-12:.3f}'),
                 # requested by the manuscript audit, 2026-10-11
                 (key, 'nd_first_isostatic_TNm', f"{r['Fd_I']*1e-12:.3f}"),
                 # ONE nd_ridge ROW, WINDOW-AVERAGED. There were two with
                 # this name and different values: this one (+-5 km window,
                 # conventions §4.1, via balance_chain) and a POINT sample
                 # np.interp(xR, x, Fd), which is not convention-compliant
                 # and read 0.491 against 0.518 in STD. The point sample is
                 # removed (2026-10-11); the published value moves by ~5 %.
                 (key, 'nd_ridge_TNm', f"{r['Fd_R']*1e-12:.3f}"),
                 (key, 'nd_max_TNm', f'{Fd[vis].max()*1e-12:.3f}'),
                 (key, 'fb_at_ridge_TNm', f'{float(np.interp(xR, x, FB))*1e-12:.3f}'),
                 (key, 'gpe_shifted_at_ridge_TNm',
                  f'{float(np.interp(xR, x, gpe_shift))*1e-12:.3f}'),
                 (key, 'identity_max_deviation_TNm', f'{dev*1e-12:.4f}'),
                 (key, 'reference_snapshot_Myr', f'{t_myr:.1f}')]
        # the glyph is now a statement about THIS snapshot, so it is drawn
        # per model from that model's own plate-wide values at the ridge
        # column -- not once for the pair (Dan, 2026-09-21)
        gl = {k: float(np.interp(xR, x, a))
              for k, a in (('gpe', d_gpe), ('nd', d_Fd), ('fb', FB),
                           ('sxx', d_Sxx))}
        # (value, positive-means-direction, colour, symbol, linewidth)
        decomposed = [(gl['gpe'], -1, 'b', r'$\Delta\mathrm{GPE}^{*}$', 3.0),
                      (gl['nd'], +1, 'k', r'$\Delta N_D$', 1.8),
                      (gl['fb'], +1, 'red', r'$F_B$', 1.8)]
        # the fundamental form has only the two terms that appear in it
        fundamental = [(gl['sxx'], +1, 'b', r'$\Delta\bar\sigma_{xx}$', 2.0),
                       (gl['fb'], +1, 'red', r'$F_B$', 2.0)]
        for nm, names, rows_ in (('fundamental', ('dSxx', 'F_B'), fundamental),
                                 ('decomposed', ('dGPE*', 'dN_D', 'F_B'), decomposed)):
            d = ', '.join(
                f'{n} {v/1e12:+.2f} '
                f'({"right" if (v >= 0) == (pd > 0) else "left"})'
                for n, (v, pd, _, _, _) in zip(names, rows_))
            print(f'   {key} glyph @ x_R [{nm}]: {d} TN/m')
        draw_direction_glyph(ax2, fundamental)
        draw_direction_glyph(ax3, decomposed)

    # SQUARE-ROOT x AXIS (Dan, 2026-09-21). x_I sits 83 km (STD) / 100 km
    # (WAL) from the trench while the ridge is at ~2900 km, so on a linear
    # axis the non-isostatic domain -- where the entire trench pull is
    # generated -- is under 3 % of the plot width and unreadable. sqrt(x)
    # is continuous and monotone, so no break is needed and the curves stay
    # single; it moves x_I to ~18 % of the width. The cost is that GRADIENTS
    # ARE NOT COMPARABLE ACROSS THE AXIS (F_B's constant slope appears to
    # flatten seaward), which is why the axis label names the scale.
    for ax in axes.flat:
        ax.set_xscale('function',
                      functions=(lambda a: np.sqrt(np.clip(a, 0, None)),
                                 lambda a: np.clip(a, 0, None) ** 2))
        ax.set_xticks(SQRT_TICKS_KM)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    # data-driven axis limits, shared across the two model columns per row
    for row, arrs in lims.items():
        lo = min(a.min() for a in arrs)
        hi = max(a.max() for a in arrs)
        pad = 0.10 * (hi - lo)
        for c in (0, 1):
            if row == 0:
                axes[row, c].set_ylim(hi + pad, lo - pad)   # w positive down
            elif row == 3:
                # SYMMETRIC about zero (Dan, 2026-09-25): the panel is
                # labelled tension-like above and compression-like below, so
                # unequal halves would misrepresent which way N_D leans.
                m = max(abs(lo), abs(hi)) + pad
                axes[row, c].set_ylim(-m, m)
            else:
                axes[row, c].set_ylim(lo - pad, hi + pad)

    # TOPOGRAPHY, not "w (positive downward)". The legend in this row says
    # "surface topography", so one word links the axis to the curve and the
    # sign convention lives in the caption (Dan, 2026-10-15).
    axes[0, 0].set_ylabel('Topography  $w$  [m]', fontsize=12)
    # 'HORIZONTAL FORCE', not bare 'Force' (Dan, 2026-10-15): every term in
    # these two rows is a force per unit distance acting in x, and saying so
    # separates them from row 4 by direction as well as by what they act on.
    axes[1, 0].set_ylabel('Horizontal force  [TN/m]', fontsize=12)
    axes[2, 0].set_ylabel('Horizontal force  [TN/m]', fontsize=12)
    # ROWS 2-3 ARE FORCES ON THE PLATE SEGMENT from x_T to x; ROW 4 IS A
    # SINGLE COLUMN'S RESULTANT (Dan, 2026-10-15). All three were labelled
    # 'Force [TN/m]', which read as one quantity in three panels.
    axes[3, 0].set_ylabel('Column resultant  [TN/m]', fontsize=12)
    # The row-4 legend is NOT optional (Dan, 2026-09-25): the y label says
    # only 'Force per unit distance', so nothing else on the panel
    # identifies which resultant this is -- and the point of the row is
    # that it is N_D ABSOLUTE, not one of the Delta forms above it. Placed
    # upper left, clear of the curve in both models and of the two words.
    # Those words carry the sign reading, not the identification.
    # "-like" on BOTH halves, symmetrically: N_D is a normal-stress
    # DIFFERENCE, so a positive value means sigma_xx exceeds sigma_zz, not
    # that the plate is in tension, and the same caveat applies in reverse.
    # SYMBOLOGY §4.4 already locks "tension-like" for exactly this reason;
    # "compression-like" is its symmetric partner.
    # LOWER RIGHT, not lower left: WAL's N_D dives to -1.9 TN/m at small x,
    # which is exactly where a lower-left legend sits. Both panels are empty
    # at the lower right.
    axes[3, 0].legend(loc='lower right', fontsize=11)
    axes[3, 1].legend(loc='lower right', fontsize=11)
    for c in (0, 1):
        for lab, dy, va in (('TENSION-LIKE', 0.94, 'top'),
                            ('COMPRESSION-LIKE', 0.06, 'bottom')):
            axes[3, c].text(0.5, dy, lab, transform=axes[3, c].transAxes,
                            # NOT bold, and not a different size from every
                            # other in-panel label. These were 11.5 pt bold at
                            # 0.80 grey -- a second typographic register for
                            # one more annotation, and too faint to read with
                            # it (Dan, 2026-10-15).
                            ha='center', va=va, fontsize=11,
                            color='0.62', zorder=0)
    if SHOW_W_TAU:
        axes[0, 0].legend(loc='lower right', fontsize=10.5, framealpha=0.9)
    axes[1, 0].legend(loc='lower left', fontsize=11)
    axes[2, 0].legend(loc='lower right', fontsize=10, ncol=2)
    fig.tight_layout()
    out = os.path.join(ROOT, 'figures', 'fig_balance_snapshot_absolute.png')
    fig.savefig(out, bbox_inches='tight', dpi=220)
    print('written:', out)
    tab = write_table('balance_snapshot_absolute', rows[0], rows[1:],
                      script='fig_balance_snapshot_absolute.py',
                      figure='fig_balance_snapshot_absolute.png',
                      models=('STD', 'WAL'),
                      meta={'identity': 'N_D(x) = N_D(x_T) + dGPE*(x) - F_B(x); '
                                        'the plotted gap is F_B',
                            'reference': 'row 4 is ABSOLUTE; rows 2-3 are Delta '
                                         'forms zeroed at the trench'})
    print('written:', tab)

if __name__ == '__main__':
    main()
