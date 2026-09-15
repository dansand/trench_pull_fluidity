"""fig_balance_snapshot — the trailing-plate force balance at the
reference snapshot (t = 40 Myr, conventions §4b mid-run rule).

Writes figures/fig_balance_snapshot.png. Reads the archive directly
(one snapshot per model); computation lifted from
notebooks/fluidity_single_step.ipynb §§7–8 (single implementation of the
resultants; the notebook remains freely re-runnable at any step).

Layout: columns STD | WAL; rows:
  1. topography w (positive downward), x_T / x_I / x_R marked
  2. the fundamental form: Δσ̄_xx(x) against −F_B(x) — the vertically
     integrated horizontal balance before any decomposition
  3. the decomposed form: ΔN_D(x), ΔGPE*(x), F_B(x), and the closure
     residual ΔN_D − ΔGPE* + F_B, PINNED over the declared
     mid-subducting-plate window (conventions §2.3: x_T + 1000..2000 km;
     the removed trench-anchor constant is printed and stated in the
     caption — the near-trench noise is shown, not hidden).

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

        xkm = (x - xT) / 1e3
        span = (xkm > -400) & (x <= xR + 200e3)
        marks = [0, (x[iI] - xT) / 1e3, (xR - xT) / 1e3]
        ax = axes[0, col]
        ax.plot(xkm[span], -fs_top[span], 'k-', lw=1.3)
        ax.axhline(0, color='0.8', lw=0.6)
        ax.set_ylim(2600, -1300)
        ax.set_title(f'{key}  (t = {t_myr:.0f} Myr)', fontsize=10)
        for xm, lab in zip(marks, ['$x_T$', '$x_I$', '$x_R$']):
            ax.axvline(xm, color='0.6', lw=0.6, ls=':')
            ax.text(xm, -1150, lab, fontsize=8, ha='center', color='0.35')
        ax = axes[1, col]
        ax.plot(xkm[span], d_Sxx[span] / 1e12, 'k-', lw=1.5,
                label=r'$\Delta\bar{\sigma}_{xx}$')
        ax.plot(xkm[span], -FB[span] / 1e12, '--', color='0.45', lw=1.3, label=r'$-F_B$')
        ax.axhline(0, color='0.8', lw=0.6)
        ax = axes[2, col]
        ax.plot(xkm[span], d_Fd[span] / 1e12, 'k-', lw=1.5, label=r'$\Delta N_D$')
        ax.plot(xkm[span], d_gpe[span] / 1e12, 'k--', lw=1.4,
                label=r'$\Delta\mathrm{GPE}^*$')
        ax.plot(xkm[span], FB[span] / 1e12, '-', color='0.45', lw=1.3, label=r'$F_B$')
        ax.plot(xkm[span], res_pin[span] / 1e12, '-', color='0.75', lw=1.0,
                label='residual (pinned)')
        ax.axhline(0, color='0.8', lw=0.6)
        ax.set_xlabel('Distance from trench [km]')
        for row in (1, 2):
            for xm in marks:
                axes[row, col].axvline(xm, color='0.6', lw=0.6, ls=':')
        del v, g

    axes[0, 0].set_ylabel('$w$ [m] (positive downward)', fontsize=9)
    axes[1, 0].set_ylabel('Force per unit\ndistance [TN/m]', fontsize=9)
    axes[2, 0].set_ylabel('Force per unit\ndistance [TN/m]', fontsize=9)
    axes[1, 0].legend(fontsize=8, frameon=False, loc='lower right')
    axes[2, 0].legend(fontsize=8, frameon=False, loc='lower right', ncol=2)
    fig.suptitle('Trailing-plate force balance at the reference snapshot: fundamental form '
                 '(middle) and decomposed form (bottom)', fontsize=10.5)
    fig.tight_layout()
    out = os.path.join(ROOT, 'figures', 'fig_balance_snapshot.png')
    fig.savefig(out, dpi=200)
    print('written:', out)

if __name__ == '__main__':
    main()
