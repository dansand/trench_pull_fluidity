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
Δσ̄_xx blue, sum green dashed) and §8.2/§9.3 (absolute form: N_D black,
[ΔGPE* + N_D(x_T)] blue thick, residual green dashed), topography panel
with the x_T (navy) / x_I / x_R annotations. SEAWARD SIDE ONLY
(xlim −50..3500 km) — the landward side is distracting (Dan).

Layout: columns STD | WAL; rows: topography · fundamental form ·
absolute (decomposed) form. The conventions §2.3 pinning constant is
computed and PRINTED (window x_T + 1000..2000 km) but the curves are the
notebook's own — the residual is displayed trench-anchored as developed.

All Δ curves are trench-referenced (±5 km window means, conventions
§2.1/§4.1). Sign pin: ΔGPE* at x_I must reproduce the committed
time-evolution trench pulls at t = 40 (1.98 STD / 1.75 WAL TN/m) — asserted.

DRAFT CAPTION. The trailing-plate force balance at the reference
snapshot (t = 40 Myr) for STD (left) and WAL (right). Top: surface
topography, with the trench, first isostatic and ridge columns marked.
Middle: the fundamental form of the vertically integrated balance — the
change in the vertically integrated horizontal normal stress, Δσ̄_xx,
against the accumulated basal traction −F_B. Bottom: the decomposed
form, ΔN_D − ΔGPE* + F_B = 0: the topographic pressure term ΔGPE*
carries the balance; ΔN_D is secondary; the residual (thin grey), with
its trench-anchor constant removed over the declared
mid-subducting-plate window, shows where extraction is imperfect —
principally the trench zone.
"""
import os, sys, glob
import numpy as np
import natsort
import pyvista as pv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cerpa_helpers import (make_field_extractor, mirror_fields_in_x, pick_trench_3step,
                           find_first_isostatic_column, find_ridge_x)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.expanduser('~/DATA/numerical_models/OUTPUTS/')
DX, ZC, Y, W = 1000.0, 75e3, 2_900_000.0, 5
T_REF_MYR = 40.0                      # conventions §4b (mid-run rule)
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

def main():
    if not hasattr(np, 'trapz'):
        np.trapz = np.trapezoid
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 8.6), sharex=True,
                             gridspec_kw={'height_ratios': [1, 1.4, 1.8]})
    for col, key in enumerate(('STD', 'WAL')):
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

        # notebook §8.2/§9.3 absolute-form quantities
        Fd_T = ca(Fd, ti)                       # N_D(x_T) — scalar offset
        bracket_term = d_gpe + Fd_T             # [ΔGPE* + N_D(x_T)]
        residual = Fd - bracket_term + FB       # the notebook's §8.3 closure

        # --- render: lifted from the notebook cells (§8.1b, §8.2/§9.3) ---
        xkm = (x - xT) / 1e3
        xi_km, xr_km = (x[iI] - xT) / 1e3, (xR - xT) / 1e3

        ax1 = axes[0, col]
        ax1.plot(xkm, -fs_top, color='k', lw=1.5, label='$w$')
        ax1.axhline(0, color='k', lw=0.5)
        for xc in (0, xi_km, xr_km):
            ax1.axvline(xc, color='k', lw=0.5)
        ax1.set_ylim(3000, -1500)
        ax1.set_title(f'{key},  $t = {t_myr:.1f}$ Myr\ntrailing-plate force balance',
                      fontsize=12)
        ax1.text(xi_km + 60, 1700,
                 'first isostatic\ncolumn (' + r'$x_I$' + ')\n' + r'$dV/dx = 0$',
                 fontsize=9)
        for x_col, lab, c in [(0, r'$x_T$', '#002147'), (xi_km, r'$x_I$', 'k'),
                              (xr_km, r'$x_R$', 'k')]:
            ax1.annotate(lab, xy=(x_col, 1), xycoords=('data', 'axes fraction'),
                         xytext=(5, -4), textcoords='offset points',
                         ha='left', va='top', color=c, fontsize=11, fontweight='bold')

        ax2 = axes[1, col]
        ax2.plot(xkm, FB * 1e-12, color='red', lw=2, label=r'$F_B(x)$')
        ax2.plot(xkm, d_Sxx * 1e-12, color='b', lw=2, label=r'$\Delta\bar\sigma_{xx}(x)$')
        ax2.plot(xkm, (d_Sxx + FB) * 1e-12, color='g', ls='--', lw=3,
                 label=r'$\Delta\bar\sigma_{xx}(x) + F_B(x)$')
        ax2.axhline(0, color='k', lw=0.5)
        for xc in (0, xi_km, xr_km):
            ax2.axvline(xc, color='k', lw=0.5)
        ax2.set_ylim(-2.5, 2.5)

        ax3 = axes[2, col]
        ax3.plot(xkm, residual * 1e-12, color='g', ls='--', lw=3,
                 label=r'$N_D(x) - [\Delta\mathrm{GPE}^{*}(x) + N_D(x_T)] + F_B(x)$')
        ax3.plot(xkm, FB * 1e-12, color='red', lw=2, label=r'$F_B(x)$')
        ax3.plot(xkm, bracket_term * 1e-12, color='b', lw=4, alpha=0.6,
                 label=r'$\Delta\mathrm{GPE}^{*}(x) + N_D(x_T)$')
        ax3.plot(xkm, Fd * 1e-12, color='k', ls='-', lw=1.5, label=r'$N_D(x)$')
        ax3.axhline(0, color='k', lw=0.5)
        for xc in (0, xi_km, xr_km):
            ax3.axvline(xc, color='k', lw=0.5)
        ax3.set_ylim(-1.5, 2.5)
        ax3.set_xlabel('Distance from trench [km]', fontsize=11)
        ax3.set_xlim(-50, 3500)                 # SEAWARD ONLY (Dan, 2026-09-15)
        del v, g

    axes[0, 0].set_ylabel('$w$ [m] (positive downward)', fontsize=10)
    axes[1, 0].set_ylabel('Force per unit distance [TN/m]', fontsize=10)
    axes[2, 0].set_ylabel('Force per unit distance [TN/m]', fontsize=10)
    axes[1, 0].legend(loc='lower right', fontsize=8)
    axes[2, 0].legend(loc='lower right', fontsize=7, ncol=2)
    fig.tight_layout()
    out = os.path.join(ROOT, 'figures', 'fig_balance_snapshot.png')
    fig.savefig(out, bbox_inches='tight', dpi=220)
    print('written:', out)

if __name__ == '__main__':
    main()
