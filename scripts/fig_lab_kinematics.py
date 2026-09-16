"""fig_lab_kinematics — what the sigma_zz zero crossing means kinematically.

Writes figures/fig_lab_kinematics.png from two committed caches:
column_profiles_deep.npz (the ridge-minus-x_I stress anomaly) and
lab_kinematics.npz (velocity, strain rate and shear stress averaged
across the trailing plate, x_I to the ridge). Build both first.

Four panels, mid-run average (36–44 Myr), models overlaid in the brand
colours per FIGURE_STYLE.md:
  1. sigma_zz anomaly (ridge − x_I), pressure register — the zero
     crossing (dotted) is the candidate base-of-lithosphere level
  2. v_x − v_plate — the dynamical LAB in the sense of Garel et al.:
     the base of the coherently translating ("constant-velocity") plate
  3. log10 strain-rate invariant — plate interior vs asthenosphere; the
     x-range excludes zero deliberately (the structure lives at
     10^-19..10^-14, Dan 2026-09-16)
  4. d<tau_zx>/dz against the independently measured plate-wide
     dP/dx (dashed). Averaging horizontal momentum across the span gives
     d<tau_zx>/dz = −Δσ̄_xx/span, so in the channel these must agree —
     they do, to 7–15 %. ABOVE the crossing the plate average is
     dominated by lithospheric flexure near x_I and is plotted faint:
     that part of the curve is not the channel signal.

What it establishes (2026-09-16): the crossing sits inside the
lithosphere–asthenosphere transition, just above the dynamical LAB, where
strain rates are already within an order of magnitude of asthenospheric
values — i.e. at the base of the coherently translating plate. Above it
the topographic pressure gradient drives the plate; below it the same
gradient drives counterflow, carried by the vertical gradient of the
shear stress.

DRAFT CAPTION. The trailing plate averaged from the first isostatic
column to the ridge, mid-run (36–44 Myr), for STD (navy) and WAL
(magenta). (a) Vertical normal stress anomaly of the ridge column
relative to the first isostatic column; its zero crossing (dotted) marks
the change from a topographic pressure gradient that drives the plate to
an adverse gradient that drives return flow. (b) Horizontal velocity
relative to the plate: the crossing lies just above the base of the
coherently translating plate. (c) Strain-rate invariant: at the crossing
the material already deforms at rates approaching asthenospheric values.
(d) Vertical gradient of the plate-averaged shear stress; below the
crossing it matches the independently measured plate-wide pressure
gradient (dashed), confirming that the deep anomaly is the channel
pressure gradient. Faint where lithospheric flexure dominates the average.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import column_profiles_cache as cpc
import lab_kinematics_cache as lkc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = {'STD': '#002147', 'WAL': '#E5007D'}
C_RULE = '#BFC3D1'
LAB_FRAC = 0.10            # dynamical-LAB criterion: |v − v_plate| > 10 % of v_plate

def crossing_depth(p_r, zkm, zmin_km=20.0):
    s = np.where((p_r[:-1] > 0) & (p_r[1:] <= 0) & (zkm[:-1] > zmin_km))[0]
    return zkm[s[0]] if len(s) else np.nan

def main():
    d, k = cpc.load(), lkc.load()
    zk = k['z'] / 1e3
    fig, axes = plt.subplots(1, 4, figsize=(14, 5.8), sharey=True)
    for key in ('STD', 'WAL'):
        col = C[key]
        c = cpc.derive(d, key)
        m = c['mid']
        zp = c['z'] / 1e3
        p_r = gaussian_filter1d(c['p_R'][m].mean(axis=0), 2) / 1e6      # MPa
        z_cross = crossing_depth(p_r, zp)
        km = (k[f'{key}_t'] >= lkc.MIDRUN_MYR[0]) & (k[f'{key}_t'] <= lkc.MIDRUN_MYR[1])
        vx = k[f'{key}_vx'][km].mean(axis=0)
        eii = k[f'{key}_log_eii'][km].mean(axis=0)
        txz = gaussian_filter1d(k[f'{key}_txz'][km].mean(axis=0), 2)
        span = k[f'{key}_span'][km].mean()
        v_plate = vx[zk < 20].mean()
        v_rel = vx - v_plate
        j = np.where(np.abs(v_rel) > LAB_FRAC * abs(v_plate))[0]
        z_lab = zk[j[0]] if len(j) else np.nan
        dtxz = np.gradient(txz, k['z'])                                 # Pa/m
        dPdx = c['dP'][m].mean() / span                                 # Pa/m, measured

        axes[0].plot(p_r, zp, color=col, lw=1.9, label=key)
        axes[1].plot(v_rel, zk, color=col, lw=1.9, label=key)
        axes[2].plot(eii, zk, color=col, lw=1.9)
        above = zk <= z_cross
        axes[3].plot(dtxz[above], zk[above], color=col, lw=1.4, alpha=0.25)
        axes[3].plot(dtxz[~above], zk[~above], color=col, lw=1.9)
        axes[3].axvline(dPdx, color=col, lw=1.2, ls='--')
        for ax in axes:
            ax.axhline(z_cross, color=col, lw=0.9, ls=':')
        ch = (zk >= 120) & (zk <= 220)
        print(f'{key}: sigma_zz crossing {z_cross:.0f} km | dynamical LAB {z_lab:.0f} km | '
              f'v_plate {v_plate:+.2f} cm/yr | log10 eII: plate {eii[zk < 60].min():.1f}, '
              f'crossing {np.interp(z_cross, zk, eii):.1f}, asth {eii[(zk > 150) & (zk < 250)].mean():.1f}')
        print(f'   d<tau_zx>/dz (120–220 km) {dtxz[ch].mean():+.2f} Pa/m vs measured dP/dx '
              f'{dPdx:+.2f} Pa/m  -> ratio {dtxz[ch].mean() / dPdx:.2f}')

    for ax, lab in zip(axes, [r'$\sigma_{zz}$ anomaly, ridge $-\,x_I$ [MPa]',
                              r'$v_x - v_{\rm plate}$ [cm/yr]',
                              r'$\log_{10}\dot\varepsilon_{II}$ [s$^{-1}$]',
                              r'$\partial\langle\tau_{zx}\rangle/\partial z$ [Pa/m]'
                              '\n(dashed: measured $\\partial P/\\partial x$)']):
        ax.set_xlabel(lab, fontsize=10)
        ax.grid(alpha=0.2, color=C_RULE, lw=0.6)
    # bold zero lines — but NOT on the strain-rate panel (its structure
    # lives far from zero; including the axis squashes it)
    for ax in (axes[0], axes[1], axes[3]):
        ax.axvline(0, color='k', lw=1.4)
    axes[2].set_xlim(-19.5, -14.2)
    axes[3].set_xlim(-6, 6)
    axes[0].set_ylabel('Depth [km]', fontsize=11)
    axes[0].set_ylim(250, 0)
    axes[0].legend(frameon=False, fontsize=10)
    fig.suptitle('Beneath the trailing plate ($x_I$ to ridge, mid-run average): '
                 'the stress anomaly and the kinematics\n'
                 '(dotted: $\\sigma_{zz}$ zero crossing — the candidate base of the '
                 'coherently translating plate)', fontsize=11)
    fig.tight_layout()
    out = os.path.join(ROOT, 'figures', 'fig_lab_kinematics.png')
    fig.savefig(out, bbox_inches='tight', dpi=220)
    print('written:', out)

if __name__ == '__main__':
    main()
