"""fig_balance_snapshot — the trailing-plate force balance at the
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
trench is plotted on a square-root scale, which expands the
non-isostatic domain between the trench and the first isostatic column;
gradients are therefore not comparable along the axis. Top: surface
topography, with the trench, first isostatic and ridge columns marked.
Middle: the fundamental form of the vertically integrated balance — the
change in the vertically integrated horizontal normal stress, Δσ̄_xx,
against the accumulated basal traction −F_B. Bottom: the decomposed
form, ΔN_D − ΔGPE* + F_B = 0: the topographic pressure term ΔGPE*
carries the balance; ΔN_D is secondary; the residual (thin grey), with
its trench-anchor constant removed over the declared
mid-subducting-plate window, shows where extraction is imperfect —
principally the trench zone. The arrows in both force panels give the
direction in which each term acts across the trailing plate at this time
step: in the fundamental form Δσ̄_xx drives the plate trench-ward against
the basal traction, and in the decomposed form ΔGPE* drives while at
this epoch both ΔN_D and the accumulated basal traction resist. ΔGPE*
acts trench-ward and F_B seaward throughout both runs; ΔN_D is the term
that changes sign, and it is shown as measured at the plotted step.
"""
import os, sys, glob
import numpy as np
import natsort
import pyvista as pv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fluidity_helpers import (make_field_extractor, mirror_fields_in_x, pick_trench_3step,
                           find_first_isostatic_column, find_ridge_x)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.expanduser('~/DATA/numerical_models/OUTPUTS/')
DX, ZC, Y, W = 1000.0, 75e3, 2_900_000.0, 5
T_REF_MYR = 40.0                      # conventions §4b (mid-run rule)
TOPO_SMOOTH_KM = 10.0                 # display smoothing of w (short-wavelength noise)
# sqrt-x tick set: round numbers chosen to be near-EVENLY spaced once
# square-rooted (0, 7.1, 14.1, 22.4, 31.6, 44.7, 54.8), so the labels do
# not collide near the trench the way a linear-looking set does
SQRT_TICKS_KM = [0, 50, 200, 500, 1000, 2000, 3000]
PIN_KM = (1000.0, 2000.0)             # closure pinning window rel. x_T (conventions §2.3)
COMMITTED_TP_T40 = {'STD': 1.98e12, 'WAL': 1.75e12}   # N/m, time-evolution cache at t=40

def load_snapshot(key):
    for ff in natsort.natsorted(glob.glob(DATA + f'{key}_RefModel/subduction_with_LM*.pvtu')):
        v = pv.read(ff)
        t = float(np.asarray(v['NormalSP::Time']).flat[0]) / 31557600.0 / 1e6
        if abs(t - T_REF_MYR) < 1.0:
            return v, t
        del v
    raise SystemExit(f'{key}: no snapshot within 1 Myr of t = {T_REF_MYR}')

def draw_direction_glyph(ax, terms, x0=0.365, y_top=1.0):
    """Force-direction glyph for THE SNAPSHOT BEING PLOTTED.

    `terms` is a list of (value, positive_means, colour, symbol, lw), one
    row per term, drawn top-down. `positive_means` is +1 if a positive
    value of that term is a force to the RIGHT (seaward) and -1 if it is
    a force to the LEFT (trench-ward); the arrow flips when the value is
    negative. See the module docstring for the per-term conventions.

    x0 is an AXES FRACTION, so on the sqrt scale 0.365 is ~465 km from
    the trench -- moved out from ~315 km (Dan, 2026-09-21) because the
    header and the GPE* label were running across the x_I line.
    """
    L = 0.09
    ax.text(x0, y_top, 'trailing-plate force balance\n(current time step)',
            transform=ax.transAxes, fontsize=9, style='italic',
            color='0.25', ha='center', va='top', linespacing=1.3)
    # Header clearance and row spacing are set in INCHES and converted to
    # this axes' own fraction. The rows sit in different-height panels
    # (height_ratios 1 : 1.4 : 1.8) while the text stays a fixed point
    # size, so a single hard-coded fraction that clears the header in the
    # tall panel runs straight through it in the short one.
    h_in = ax.get_position().height * ax.get_figure().get_figheight()
    y0 = y_top - 0.34 / h_in                 # first row, below the header
    ys = [y0 - (0.17 / h_in) * i for i in range(len(terms))]
    ax.plot([x0, x0], [min(ys) - 0.02, max(ys) + 0.02], color='0.5', lw=0.8,
            transform=ax.transAxes)
    for y, (val, pos_dir, c, lab, lw) in zip(ys, terms):
        s = pos_dir if val >= 0 else -pos_dir      # the actual direction
        ax.annotate('', xy=(x0 + s * L, y), xytext=(x0, y),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='-|>', color=c, lw=lw,
                                    mutation_scale=14))
        ax.text(x0 + s * (L + 0.015), y, lab, transform=ax.transAxes,
                fontsize=9, color=c, va='center',
                ha='right' if s < 0 else 'left')


def compute(key):
    """Every term of the trailing-plate balance at the reference snapshot.

    Single implementation, shared by fig_balance_snapshot (the Delta form)
    and fig_balance_absolute (the form shifted by N_D(x_T), which shows
    the actual N_D rather than its change). Returns a dict of arrays in
    the analysis frame.
    """
    if not hasattr(np, 'trapz'):
        np.trapz = np.trapezoid
    v, t_myr = load_snapshot(key)
    v.point_data['p'] = v['NormalSP::Pressure']
    v.point_data['tzz'] = v['NormalSP::Stress'][:, 4]
    v.point_data['txx'] = v['NormalSP::Stress'][:, 0]
    v.point_data['txz'] = -v['NormalSP::Stress'][:, 1]
    v.point_data['T'] = v['NormalSP::Temperature']
    v.point_data['fs'] = v['NormalSP::FreeSurface']
    vel = v['NormalSP::Velocity'] / 3.17098e-10
    v.point_data['vx'] = vel[:, 0]; v.point_data['vy'] = vel[:, 1]
    x0, x1, _, _, _, _ = v.bounds
    nx, nz = int((x1 - x0) / DX), int(ZC / DX)
    x = x0 + (np.arange(nx) + 0.5) * DX
    z = (np.arange(nz) + 0.5) * DX
    X, Z = np.meshgrid(x, z, indexing='xy')
    g = pv.StructuredGrid(X.T, (Y - Z).T, np.zeros_like(X.T)).sample(v)
    nxp, nzp, _ = g.dimensions
    val = g['vtkValidPointMask'].reshape((nxp, nzp), order='F').astype(bool)
    gf = make_field_extractor(g, nxp, nzp, val)
    p, txx, tzz, txz = gf('p'), gf('txx'), gf('tzz'), gf('txz')
    T, fs, vx, vz = gf('T'), gf('fs'), gf('vx'), -gf('vy')
    p, txx, tzz, txz, T, fs, vx, vz = mirror_fields_in_x(p, txx, tzz, txz, T, fs, vx, vz)
    p, txx, tzz, txz = (np.nan_to_num(a) for a in (p, txx, tzz, txz))
    fs_top = np.nan_to_num(fs[0, :])
    xT, _ = pick_trench_3step(x, z, p, vx, subducting_side='right')
    ti = int(np.argmin(np.abs(x - xT)))
    iI, _ = find_first_isostatic_column(x, fs_top, xT, ti, DX, seaward_sign=+1)
    xR, iR = find_ridge_x(x, fs_top, xT, seaward_sign=+1)
    ca = lambda f, j: f[..., max(0, j - W):j + W + 1].mean(axis=-1)

    # resultants (notebook §7 chain, verbatim logic)
    Fd = np.trapz(txx - tzz, z, axis=0)
    Sxx = np.trapz(-p + txx, z, axis=0)
    gpe = -np.trapz(-p + tzz, z, axis=0)
    tau_b = txz[-1, :]
    FB_ = cumulative_trapezoid(tau_b, x, initial=0.0)
    d_Fd = Fd - ca(Fd, ti)
    d_Sxx = Sxx - ca(Sxx, ti)
    d_gpe = gpe - ca(gpe, ti)
    FB = FB_ - ca(FB_, ti)

    # sign pin: trench pull at x_I vs the committed series
    tp = ca(d_gpe, iI)
    rel = abs(tp - COMMITTED_TP_T40[key]) / COMMITTED_TP_T40[key]
    assert rel < 0.05, f'{key}: ΔGPE*(x_I) = {tp/1e12:.2f} TN/m vs committed — chain broken'

    # pinned closure (conventions §2.3)
    res = d_Fd - d_gpe + FB
    pin = (x > xT + PIN_KM[0] * 1e3) & (x < xT + PIN_KM[1] * 1e3)
    res_const = res[pin].mean()
    res_pin = res - res_const
    print(f'{key} t={t_myr:.1f}: ΔGPE*(x_I) {tp/1e12:+.2f} TN/m (pin OK); '
          f'closure constant removed {res_const/1e12:+.3f} TN/m '
          f'(window x_T+{PIN_KM[0]:.0f}..{PIN_KM[1]:.0f} km); '
          f'rms about pinned closure {res_pin[pin].std()/1e12:.3f} TN/m')


    return dict(x=x, xT=xT, xI=x[iI], xR=xR, t_myr=t_myr, z=z,
                fs_top=fs_top, Fd=Fd, Sxx=Sxx, gpe=gpe,
                d_Fd=d_Fd, d_Sxx=d_Sxx, d_gpe=d_gpe, FB=FB,
                Fd_T=ca(Fd, ti), res_pin=res_pin, res_const=res_const)


def main():
    if not hasattr(np, 'trapz'):
        np.trapz = np.trapezoid
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 8.6), sharex=True,
                             gridspec_kw={'height_ratios': [1, 1.4, 1.8]})
    lims = {0: [], 1: [], 2: []}          # per-row plotted data, for axis limits
    for col, key in enumerate(('STD', 'WAL')):
        r = compute(key)
        x, xT, t_myr = r['x'], r['xT'], r['t_myr']
        fs_top, d_Sxx, d_gpe, d_Fd, FB = (r['fs_top'], r['d_Sxx'], r['d_gpe'],
                                          r['d_Fd'], r['FB'])
        res_pin = r['res_pin']
        iI_x, xR = r['xI'], r['xR']
        # --- render: lifted from the notebook cells (§8.1b, §8.2) ---
        xkm = (x - xT) / 1e3
        xi_km, xr_km = (iI_x - xT) / 1e3, (xR - xT) / 1e3
        vis = (xkm >= -50) & (xkm <= 3500)      # the displayed span

        # topography: display-smoothed SEAWARD OF x_I ONLY — the
        # trailing-plate short-wavelength content otherwise dominates
        # visually (Dan), but the narrow trench is REAL structure and
        # filtering shaves hundreds of metres off w_T (the aspect lesson):
        # raw through the trench zone, blended into the smoothed curve
        # over 100 km beyond x_I.
        topo_raw = -fs_top
        topo_sm = -gaussian_filter1d(fs_top, TOPO_SMOOTH_KM * 1e3 / DX)
        wgt = np.clip(((x - iI_x) / 1e3 - 50.0) / 100.0, 0.0, 1.0)
        topo_disp = topo_raw * (1 - wgt) + topo_sm * wgt

        ax1 = axes[0, col]
        ax1.plot(xkm, topo_disp, color='k', lw=1.5, label='$w$')
        ax1.axhline(0, color='k', lw=0.5)
        for xc in (0, xi_km, xr_km):
            ax1.axvline(xc, color='k', lw=0.5)
        lims[0].append(topo_disp[vis])
        ax1.set_title(f'{key},  $t = {t_myr:.1f}$ Myr\ntrailing-plate force balance',
                      fontsize=12)
        ax1.text(xi_km + 60, 0.75 * topo_disp[vis].max(),
                 'first isostatic\ncolumn (' + r'$x_I$' + ')\n' + r'$dV/dx = 0$',
                 fontsize=9)
        # x_T now sits ON the left spine (the sqrt axis starts at 0), so its
        # label must go inside the axes, not to the left of the line
        for x_col, lab, c, side in [(0, r'$x_T$', '#002147', 'left'),
                                    (xi_km, r'$x_I$', 'k', 'left'),
                                    (xr_km, r'$x_R$', 'k', 'left')]:
            ax1.annotate(lab, xy=(x_col, 1), xycoords=('data', 'axes fraction'),
                         xytext=(5 if side == 'left' else -5, -4),
                         textcoords='offset points',
                         ha=side, va='top', color=c, fontsize=11,
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
        ax3.set_xlabel('Distance from trench [km]  ' + r'($\sqrt{x}$ scale)',
                       fontsize=11)
        ax3.set_xlim(0, 3500)                   # SEAWARD ONLY (Dan, 2026-09-15)
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
            else:
                axes[row, c].set_ylim(lo - pad, hi + pad)

    axes[0, 0].set_ylabel('$w$ [m] (positive downward)', fontsize=10)
    axes[1, 0].set_ylabel('Force per unit distance [TN/m]', fontsize=10)
    axes[2, 0].set_ylabel('Force per unit distance [TN/m]', fontsize=10)
    axes[1, 0].legend(loc='lower left', fontsize=8)
    axes[2, 0].legend(loc='lower right', fontsize=7, ncol=2)
    fig.tight_layout()
    out = os.path.join(ROOT, 'figures', 'fig_balance_snapshot.png')
    fig.savefig(out, bbox_inches='tight', dpi=220)
    print('written:', out)

if __name__ == '__main__':
    main()
