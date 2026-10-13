"""fig_trench_stress_profiles — the two stress profiles at the column of
maximum bending moment, with the mechanical-thickness estimates marked.

Writes figures/fig_trench_stress_profiles.png from the committed
column-profile cache. Drawn at the reference epoch, averaged over the
mid-run window (36-44 Myr, conventions §4b) rather than a single
snapshot: the profiles are noisy at the trench and the average is the
stable representation.

Four panels showing how the trench pressure deficit is equilibrated
inside the strong part of the lithosphere (Dan, 2026-09-17 — the
manuscript otherwise shows nothing of the stress regime in the bending
region):
  (a) DELTA SIGMA_ZZ, first isostatic column minus trench — the pressure
      deficit whose integral is the trench pull.
  (b) TAU_ZX at the trench — the vertical shear stress that supports it;
      its integral is the shear resultant V.
  (c) TAU_ZX,X at the trench — the along-strike gradient of that shear
      stress, equivalently the EQUIVALENT DENSITY rho_hat = tau_zx,x / g
      (second axis). The companion study identifies the vertical centre
      of mass of this distribution as the effective moment arm of the
      trench pull; measured here it is 34 km (STD) / 31 km (WAL),
      against a measured conversion length of 33.9 / 33.8 km and
      h/2 = 31.5 / 27.5 km.
  (d) SIGMA_XX - SIGMA_ZZ at the column of MAXIMUM BENDING MOMENT
      (~20-25 km seaward of the trench) — the flexural envelope, whose
      zero crossing is the neutral plane and whose outer-fibre peaks
      bound the strong part of the plate.

Panels (a)-(c) are referenced to the trench, where the force balance is
taken; (d) is at the moment maximum, where the flexural signal is
strongest.

HOW tau_zx,x IS CONSTRUCTED. tau_zx is sampled on the analysis-frame
grid (1 km in x and z); smoothed ALONG X ONLY with a Gaussian of
sigma = 10 km (the raw field cannot be differentiated); central-
differenced in x at 1 km; then averaged over the +-5 km column window.
No smoothing in depth precedes the derivative, so the vertical structure
is not manufactured by the filter. Display adds a 2 km depth smooth.

SMOOTHING SENSITIVITY (STD, t = 40 Myr, trench column, 0-100 km). The
PEAK depends on the filter; the CENTRE OF MASS -- which is what the
argument rests on -- does not:

    sigma      rho_hat peak      centre of mass
    none        251 kg/m3           33.1 km
    5 km        236                 33.1
    10 km       201                 33.6   <- used
    20 km       147                 35.7
    40 km        97                 40.8

Quote the peak only with its smoothing stated, or quote the unsmoothed
value (251 kg/m3). The centre of mass is a property of the field.

TIME AVERAGING: the mid-run window 36-44 Myr (conventions 4b) -- five
snapshots for STD, four for WAL.

Overlaid on both, per model: the span of the three mechanical thickness
estimates (2 h_np, the 25 MPa fibre-stress cap, thermal
900 C) as a shaded band, with the neutral plane h_np marked separately. Reading the
two panels together is the point — the same thickness has to make sense
of the flexural envelope on the left and of the pressure deficit on the
right, and it does: the deficit dies where the envelope does.

DRAFT CAPTION. Stress profiles at the column of maximum bending moment
(about 20-25 km seaward of the trench), averaged over the mid-run window,
for STD (navy) and WAL (magenta). Left: the normal-stress difference
(flexural fibre stress); the neutral plane (dotted) is its zero crossing.
Right: the vertical normal stress difference between that column and the
first isostatic column. Shaded bands span the three estimates
of the mechanical thickness; both the flexural envelope and the pressure
deficit decay away within the same depth range.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import column_profiles_cache as cpc
from fluidity_helpers import depth_ticks
from tables_io import write_table
from fig_mechanical_thickness import thicknesses

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = {'STD': '#002147', 'WAL': '#E5007D'}
C_RULE = '#BFC3D1'
Z_TP_KM = 120.0    # declared integration depth for the trench-pull area.
                   # The deficit has converged well before it: the integral
                   # reaches 94-96 % by 60 km, 98 % by 75 km and 99 % by
                   # 90 km, and is flat from 120 to 200 km (<1 %). Stated
                   # on the figure rather than left implicit (Dan, 2026-09-16).

def main():
    d = cpc.load()
    rows = [('model', 'quantity', 'value')]
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 6.2), sharey=True)
    for key in ('STD', 'WAL'):
        col = C[key]
        c = cpc.derive(d, key)
        r = thicknesses(d, key)
        m = c['mid']
        z, zkm = c['z'], c['z'] / 1e3
        sm = lambda a: gaussian_filter1d(a, 2)
        fib = sm((d[f'{key}_sxx_M'] - d[f'{key}_szz_M'])[m].mean(axis=0)) / 1e6
        dzz = -sm(c['p_T'][m].mean(axis=0)) / 1e6            # iso - trench
        txz = sm(d[f'{key}_txz_T'][m].mean(axis=0)) / 1e6
        dtxz = sm(d[f'{key}_dtxz_T'][m].mean(axis=0))
        dx_M = (d[f'{key}_xM'][m] - d[f'{key}_xT'][m]).mean() / 1e3
        shal = z <= 100e3
        # cell rule in BOTH numerator and denominator (the ratio barely
        # moves, but one convention)
        dzf = float(np.diff(z)[0])
        arm = ((dtxz[shal] * z[shal]).sum() * dzf) / (dtxz[shal].sum() * dzf)
        axes[0].plot(dzz, zkm, color=col, lw=2.0, label=key)
        axes[1].plot(txz, zkm, color=col, lw=2.0, label=key)
        axes[2].plot(dtxz / 9.8, zkm, color=col, lw=2.0, label=key)
        axes[2].axhline(arm / 1e3, color=col, lw=1.2, ls='-.')
        axes[3].plot(fib, zkm, color=col, lw=2.0, label=key)
        # PEAKS ARE TAKEN ON |.| AND REPORTED SIGNED. tau_zx at the trench is
        # POSITIVE throughout the plate in the model's own frame, so min() --
        # which was the peak while the analysis was mirrored -- returns the
        # near-surface value instead: it tabulated +0.1 MPa at 0 km against a
        # true peak of +75 (STD) / +64 (WAL) MPa at ~30 km, which is the
        # neutral plane, where the flexural shear should be largest. The same
        # slip ran the other way in the deficit print (min() = -0.6 MPa).
        # Taking |.| is frame-proof; the sign is preserved in what is written.
        def peak(a, zm):
            j = int(np.argmax(np.abs(a[zm])))
            return float(a[zm][j]), float(zkm[zm][j])
        w80 = zkm < 80
        t_pk, t_zpk = peak(txz, w80)
        r_pk, r_zpk = peak(dtxz / 9.8, w80)
        d_pk, _ = peak(dzz, np.ones_like(zkm, bool))
        print(f'{key}: tau_zx peak {t_pk:+.0f} MPa at {t_zpk:.0f} km; rho_hat peak '
              f'{r_pk:+.0f} kg/m3; centre of mass of tau_zx,x '
              f'= {arm/1e3:.0f} km (the effective moment arm)')
        rows += [(key, 'tau_zx_peak_MPa', f'{t_pk:.1f}'),
                 (key, 'tau_zx_peak_depth_km', f'{t_zpk:.0f}'),
                 (key, 'rho_hat_peak_kgm3_smooth10km', f'{r_pk:.0f}'),
                 (key, 'rho_hat_peak_depth_km', f'{r_zpk:.0f}'),
                 (key, 'tau_zx_x_centre_of_mass_km', f'{arm/1e3:.1f}'),
                 (key, 'deficit_peak_MPa', f'{d_pk:.1f}'),
                 (key, 'max_moment_offset_km', f'{dx_M:.0f}'),
                 (key, 'x_smoothing_km', '10'),
                 (key, 'midrun_snapshots', f'{int(m.sum())}')]
        ests = np.array([r[k][m].mean() for k in ('np2', 'yield_cap', 'thermal')]) / 1e3
        h_np = r['np2'][m].mean() / 2e3
        for ax in axes:
            ax.axhspan(ests.min(), ests.max(), color=col, alpha=0.10, lw=0)
            ax.axhline(np.median(ests), color=col, lw=1.0, ls='--')
        axes[3].axhline(h_np, color=col, lw=1.0, ls=':')
        axes[3].text(0.97, h_np - 1.5, f'$h_{{np}}$ {h_np:.0f} km', color=col,
                     fontsize=8, ha='right', transform=axes[3].get_yaxis_transform())
        print(f'{key}: max-|M| column at x_T{dx_M:+.0f} km; h_np {h_np:.0f} km; estimates {ests.min():.0f}-{ests.max():.0f} km '
              f'(median {np.median(ests):.0f}); fibre peaks '
              f'{fib[zkm < h_np].max():+.0f} / {fib[(zkm > h_np) & (zkm < 120)].min():+.0f} MPa; '
              f'deficit peak {d_pk:+.0f} MPa')
    axes[0].set_xlabel(r'(a) $\Delta\sigma_{zz}$, $x_I-$trench [MPa]', fontsize=10.5)
    axes[1].set_xlabel(r'(b) $\tau_{zx}$ at the trench [MPa]', fontsize=10.5)
    axes[2].set_xlabel(r'(c) $\hat\rho = \tau_{zx,x}/g$ [kg m$^{-3}$]'
                       '\n(dash-dot: centre of mass)', fontsize=10.5)
    axes[3].set_xlabel(r'(d) $\sigma_{xx}-\sigma_{zz}$ at max $M$ [MPa]',
                       fontsize=10.5)
    axes[0].set_ylabel('Depth [km]', fontsize=11)
    axes[0].set_ylim(Z_TP_KM, 0)

    for ax in axes:
        ax.axvline(0, color='k', lw=1.6)
        ax.grid(alpha=0.25, color=C_RULE, lw=0.6)
        ax.legend(frameon=False, fontsize=8.5, loc='lower right') if ax is axes[0] else None
    fig.suptitle('Equilibration of the trench pressure deficit', fontsize=11)
    fig.tight_layout()
    out = os.path.join(ROOT, 'figures', 'fig_trench_stress_profiles.png')
    depth_ticks(fig)   # 20 km depth ticks (FIGURE_STYLE)
    fig.savefig(out, bbox_inches='tight', dpi=220)
    print('written:', out)
    tab = write_table('bending_stress_regime', rows[0], rows[1:],
                      script='fig_trench_stress_profiles.py', figure='fig_trench_stress_profiles.png',
                      models=('STD', 'WAL'), meta={'x_smoothing_km': 10, 'midrun_window_Myr': list(cpc.MIDRUN_MYR)})
    print('written:', tab)

if __name__ == '__main__':
    main()
