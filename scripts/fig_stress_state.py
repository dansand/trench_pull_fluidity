"""fig_stress_state — the trench stress state, and how much the plane matters.

Writes figures/fig_stress_state.png and tables/stress_state.csv from
stress_section_cache.py (the maps) and trench_plane_cache.py (the time
series). Build both first.

SI FIGURE (Dan, 2026-10-09). The manuscript tests the slab-pull orthodoxy
with the normal-stress-difference resultant on a VERTICAL plane at the
trench. The plate is not quite horizontal there, so the honest question is
how much of the answer that plane choice carries.

  (a), (b)  the normal-stress difference around the trench at the reference
            epoch, with the three key columns, the isotherms, the vertical
            integration region (dashed, to z_c = 80 km) and the two planes
  (c)       the trench resultant on both planes through the whole run
  (d), (e)  the two epochs where the stress state is unambiguous -- STD
            early, when the trench column is tension-like, and WAL late,
            when it is strongly compression-like -- as integrand profiles

HOW THE COMPARISON IS SCALED, and why the obvious scaling is wrong.
Dan's ruling (2026-10-09): do NOT quote the shift as a percentage of the
instantaneous resultant. STD's trench resultant passes through zero near
t = 35 Myr, so near the reference epoch any finite shift reads as a
runaway percentage -- an artefact of dividing by a small number, not a
measure of anything. The shift is quoted against the run-scale amplitude
of N_D(x_T) (its largest and median absolute value) and against the trench
pull it is being compared with.

THE ALGEBRA, which is the whole result:

    N_D^tilt - N_D = (cos 2theta - 1) N_D + 2 sin 2theta * V

The first term is the dip acting on the resultant itself and is ~1 % at
4 degrees. The second is the dip acting on the SHEAR LOAD
V = int tau_zx dz, which at the trench is 0.8-2.0 TN/m. The plane choice
therefore moves the trench resultant by a few tenths of a TN/m almost
irrespective of what the resultant is -- it is a shear-load term, not a
dip term. "The plate dips by only a few degrees" is true and is NOT
sufficient on its own; what carries the argument is that the shift is
small against the forces being compared and never changes the sign of the
trench resultant except within a few tenths of a TN/m of zero.

WHICH WAY THE CORRECTION GOES, and why that settles it. V at the trench is
POSITIVE at every snapshot of both runs, so the shift 2 sin 2theta * V is
positive at every snapshot too: the slab-normal plane always reads MORE
tension-like than the vertical one. The correction therefore runs in the
orthodoxy's favour -- and is still not enough. The trench column stays
compression-like on the slab-normal plane at 36/37 STD snapshots and
37/37 WAL snapshots; the single exception is STD t = 36 Myr, where the
vertical resultant is -0.26 TN/m, 11 % of the run's largest.

⚠ SYMBOL BLOCKER, recorded rather than worked around. SYMBOLOGY §7.1: the
slab-frame coordinate pair is unruled upstream. The interim rule is
followed -- the rotation is written in x, z components and theta, with no
slab-frame subscripts.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import RegularGridInterpolator

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import column_profiles_cache as cpc
import stress_section_cache as ssc
import trench_plane_cache as tpc
from fluidity_helpers import depth_ticks
from tables_io import write_table

ROOT = os.path.dirname(_HERE)
VMAX_MPA = 300.0
ZC_KM = cpc.ZC_KM               # conventions §5.1
PROFILE_ZMAX_KM = 115.0       # (d), (e) are drawn below z_c deliberately
XLIM_KM = (-60.0, 200.0)      # seaward distance
ZLIM_KM = (150.0, -8.0)
C = {'STD': '#002147', 'WAL': '#E5007D'}
C_RULE = '#BFC3D1'
DS = 250.0                    # step along the tilted path [m]


def resultants(ns, tzx, z, x_rel, dip, w_km=5.0, zc_km=ZC_KM):
    """N_D(x_T) to z_c on the vertical plane and on the slab-normal plane.

    ns, tzx are (nx, nz) patches; x_rel is x - x_T. Columns are +/-w_km
    window means (conventions §4.1). Returns (vertical, rotation-only on
    the vertical trace, tilted path to the same depth, V).
    """
    zc = zc_km * 1e3
    wz = z <= zc
    wx = np.abs(x_rel) <= w_km * 1e3
    nsc, tzc = ns[wx].mean(axis=0), tzx[wx].mean(axis=0)
    # VERTICAL PATH: the shared cell rule, 0 -> z_c. The patch grid is
    # cell-centred at 0.25 km on a 0.5 km spacing, so it is boundary-aligned
    # and integrate_to_zc accepts it (160 cells). These were trapz over the
    # centres, i.e. 0.25 -> 79.75 km under a 0 -> 80 km label.
    nd = cpc.integrate_to_zc(nsc, z, zc)
    V = cpc.integrate_to_zc(tzc, z, zc)
    rot = cpc.integrate_to_zc(
        nsc * np.cos(2 * dip) + 2 * tzc * np.sin(2 * dip), z, zc)
    # TILTED PLANE: perpendicular to the plate, so it drifts seaward (-x in
    # the native frame) as it deepens. This is a LINE INTEGRAL along s from
    # s = 0 to s = z_c / cos(theta) and NOT a column-cell integral, so it is
    # not routed through integrate_to_zc (Codex, 2026-10-10). It is made
    # endpoint-complete the same way: s-cells that tile 0 .. L exactly,
    # evaluated at their centres, so the path covers the full plane depth
    # rather than starting at the first patch centre and stopping short of
    # z_c. Before, `arange(...)*DS` began at s = 0 but the field sample was
    # clipped to z[0] = 0.25 km, and `zs <= zc` truncated the deep end.
    L = zc / np.cos(dip)
    n_s = max(1, int(round(L / DS)))
    ds = L / n_s                       # cells tile 0..L exactly
    s = (np.arange(n_s) + 0.5) * ds    # cell centres along the path
    xs, zs = s * -np.sin(dip), s * np.cos(dip)
    keep = np.ones(n_s, bool)
    gi = lambda F: RegularGridInterpolator((x_rel, z), F, bounds_error=False,
                                           fill_value=np.nan)
    # window-average the tilted trace the same way, across the plane
    acc_ns, acc_tz = 0.0, 0.0
    offs = np.arange(-w_km, w_km + 0.001, 1.0) * 1e3
    for o in offs:                       # offset along the plate direction
        # clipped to the patch: the plane starts at the first cell centre,
        # the same depth the vertical integral starts from
        pts = np.column_stack([
            np.clip(xs[keep] + o * np.cos(dip), x_rel[0], x_rel[-1]),
            np.clip(zs[keep] + o * np.sin(dip), z[0], z[-1])])
        acc_ns = acc_ns + gi(ns)(pts); acc_tz = acc_tz + gi(tzx)(pts)
    nst, tzt = acc_ns / len(offs), acc_tz / len(offs)
    # cell rule along the path: the s-cells tile 0..L, so this is the full
    # plane integral. The shallowest sample is still clipped to the patch's
    # first centre (0.25 km) because the patch does not extend above it --
    # a 0.25 km hold, against a 0.5 km cell, in the one place the field is
    # genuinely unavailable.
    tilt = float((nst * np.cos(2 * dip)
                  + 2 * tzt * np.sin(2 * dip)).sum() * ds)
    return nd, rot, tilt, V


def series(d, k):
    z, x_rel = d['z'], d[f'{k}_x_rel']
    out = []
    for i, dip in enumerate(d[f'{k}_dip']):
        out.append(resultants(d[f'{k}_ns'][i].astype(float),
                              d[f'{k}_tzx'][i].astype(float), z, x_rel, dip))
    return np.array(out).T / 1e12          # TN/m: nd, rot, tilt, V


def integrand(d, k, i, w_km=5.0):
    z, x_rel, dip = d['z'], d[f'{k}_x_rel'], float(d[f'{k}_dip'][i])
    wx = np.abs(x_rel) <= w_km * 1e3
    ns = d[f'{k}_ns'][i].astype(float)[wx].mean(axis=0)
    tz = d[f'{k}_tzx'][i].astype(float)[wx].mean(axis=0)
    return z, ns, ns * np.cos(2 * dip) + 2 * tz * np.sin(2 * dip), dip


def main():
    d = tpc.load()
    ss = ssc.load()
    cc = np.load(os.path.join(ROOT, 'notebooks', 'outputs',
                              'column_profiles_deep.npz'))
    S = {k: series(d, k) for k in ('STD', 'WAL')}
    rows = [('model', 'quantity', 'value')]

    # the two unambiguous epochs: most tension-like (STD) and most
    # compression-like (WAL) trench column of the run
    pick = {'STD': int(np.argmax(S['STD'][0])),
            'WAL': int(np.argmin(S['WAL'][0]))}

    # LAYOUT. 14 columns: each half of the figure is 6 wide, column 6 is a
    # gutter so (d)'s axis label clears (c), and column 13 is the colourbar.
    # Both rows therefore end at the same x. The maps are equal-aspect, so
    # row 0's height is derived from the panel width and the domain aspect
    # -- otherwise the row carries a band of slack beneath them.
    ASPECT = (XLIM_KM[1] - XLIM_KM[0]) / (ZLIM_KM[0] - ZLIM_KM[1])
    FW, H1, HM = 13.0, 3.3, None
    HM = FW * 6 / 14 * 0.92 / ASPECT
    fig = plt.figure(figsize=(FW, HM + H1 + 0.95))
    gs = GridSpec(2, 14, figure=fig, height_ratios=[HM, H1],
                  hspace=0.36, wspace=1.4)
    axm = [fig.add_subplot(gs[0, 0:6]), fig.add_subplot(gs[0, 7:13])]
    axt = fig.add_subplot(gs[1, 0:6])
    axp = [fig.add_subplot(gs[1, 7:10]), fig.add_subplot(gs[1, 10:13])]
    axc = fig.add_subplot(gs[0, 13])
    im = None

    # ---- (a), (b) the maps ------------------------------------------------
    zs = ss['z']; zsk = zs / 1e3
    for c, k in enumerate(('STD', 'WAL')):
        x = ss[f'{k}_x']
        xT, xI, t = (float(ss[f'{k}_xT']), float(ss[f'{k}_xI']),
                     float(ss[f'{k}_t']))
        i40 = int(np.argmin(np.abs(cc[f'{k}_t'] - t)))
        xM = float(cc[f'{k}_xM'][i40])
        ns = ss[f'{k}_sxx'] - ss[f'{k}_szz']
        tC = ss[f'{k}_temp'] - 273.15
        iref = int(np.argmin(np.abs(d[f'{k}_t'] - t)))
        dip = float(d[f'{k}_dip'][iref])

        ax = axm[c]
        sx = (xT - x) / 1e3
        im = ax.pcolormesh(sx, zsk, (ns / 1e6).T, cmap='coolwarm',
                           shading='auto', vmin=-VMAX_MPA, vmax=VMAX_MPA,
                           rasterized=True)
        ax.contour(sx, zsk, tC.T, levels=[400, 600, 800, 1000, 1200],
                   colors='grey', linewidths=1.0)
        for xx, lab in ((0.0, '$x_T$'), ((xT - xM) / 1e3, '$x_m$'),
                        ((xT - xI) / 1e3, '$x_I$')):
            ax.axvline(xx, color='0.15', lw=1.0, ls=':')
            ax.text(xx, 2.0, f' {lab}', fontsize=10, color='k', ha='left',
                    va='top')
        ax.add_patch(plt.Rectangle((0.0, 0.0), (xT - xI) / 1e3, ZC_KM,
                                   fill=False, ec='k', lw=1.5, ls='--',
                                   zorder=6))
        ax.plot([0, 0], [zs[0] / 1e3, ZC_KM], color='white', lw=3.0, zorder=7)
        ax.plot([0, 0], [zs[0] / 1e3, ZC_KM], color='k', lw=1.5, zorder=8,
                label='vertical')
        sp = np.arange(int(ZC_KM * 1e3 / np.cos(dip) / DS) + 1) * DS
        ax.plot(sp * np.sin(dip) / 1e3, sp * np.cos(dip) / 1e3,
                color='white', lw=3.2, zorder=7)
        ax.plot(sp * np.sin(dip) / 1e3, sp * np.cos(dip) / 1e3,
                color='#00A0B0', lw=1.7, zorder=8, label='slab-normal')
        ax.plot(0, 0, 'wo', mec='k', ms=7, zorder=9)
        ax.set_xlim(*XLIM_KM); ax.set_ylim(*ZLIM_KM); ax.set_aspect('equal')
        ax.set_xlabel('Seaward distance from the trench [km]', fontsize=10)
        ax.set_title(f'({"ab"[c]})  {k}   $t$ = {t:.0f} Myr   '
                     f'$\\theta$ = {np.degrees(dip):.1f}$^\\circ$',
                     fontsize=10.5, loc='left')
        if c == 0:
            ax.set_ylabel('Depth [km]', fontsize=10)
            ax.legend(loc='lower left', frameon=False, fontsize=9)
        else:
            ax.tick_params(labelleft=False)

    cb = fig.colorbar(im, cax=axc)
    cb.set_label(r'$\sigma_{xx}-\sigma_{zz}$  [MPa]', fontsize=10)

    # ---- (c) both planes, whole run --------------------------------------
    axt.axhline(0, color='k', lw=0.7)
    axt.grid(alpha=0.25, color=C_RULE, lw=0.6)
    for k in ('STD', 'WAL'):
        t = d[f'{k}_t']; nd, rot, tilt, V = S[k]
        axt.plot(t, nd, color=C[k], lw=1.8, label=k)
        axt.plot(t, tilt, color=C[k], lw=1.4, ls='--')
        axt.plot(t[pick[k]], nd[pick[k]], 'o', color=C[k], mec='white',
                 mew=0.6, ms=7, zorder=5)
    axt.set_xlabel('Model time [Myr]', fontsize=10)
    axt.set_ylabel('$N_D(x_T)$  [TN/m]', fontsize=10)
    axt.set_title('(c)  Trench resultant on the two planes', fontsize=10.5,
                  loc='left')
    axt.legend(frameon=False, fontsize=9, ncol=2, loc='lower left')
    axt.text(0.985, 0.95, 'solid: vertical\ndashed: slab-normal',
             transform=axt.transAxes, ha='right', va='top', fontsize=8.5,
             color='0.3')

    # ---- (d), (e) the unambiguous epochs ---------------------------------
    for j, k in enumerate(('STD', 'WAL')):
        i = pick[k]
        z, iv, ir, dip = integrand(d, k, i)
        nd, rot, tilt, V = S[k][:, i]
        ax = axp[j]
        # DRAWN BELOW z_c ON PURPOSE. Cutting the profile at the
        # integration depth invites the reading that the cut hides
        # something; it is cheaper to show what lies beneath it. The
        # shading -- the resultant -- still stops at z_c.
        wz = z <= ZC_KM * 1e3
        ax.axvline(0, color='k', lw=0.7)
        ax.axhline(ZC_KM, color='0.45', lw=0.9, ls=':')
        ax.fill_betweenx(z[wz] / 1e3, 0, iv[wz] / 1e6, color=C[k], alpha=0.18,
                         lw=0)
        ax.plot(iv / 1e6, z / 1e3, color='k', lw=1.6, label='vertical')
        ax.plot(ir / 1e6, z / 1e3, color='#00A0B0', lw=1.6,
                label='slab-normal')
        ax.text(0.97, ZC_KM - 1.5, '$z_c$', transform=ax.get_yaxis_transform(),
                ha='right', va='bottom', fontsize=9, color='0.35')
        ax.set_ylim(PROFILE_ZMAX_KM, 0)
        ax.grid(alpha=0.25, color=C_RULE, lw=0.6)
        ax.set_xlabel(r'$\sigma_{xx}-\sigma_{zz}$  [MPa]', fontsize=9.5)
        state = 'tension-like' if nd > 0 else 'compression-like'
        ax.set_title(f'({"de"[j]})  {k}  $t$ = {d[f"{k}_t"][i]:.0f} Myr',
                     fontsize=10.5, loc='left')
        # the two resultants sit inside the panel: the titles ran into each
        # other once they carried the numbers (and the caption carries the
        # explanation anyway)
        ax.text(0.04, 0.035, f'{state}\n{nd:+.2f} / {tilt:+.2f} TN/m',
                transform=ax.transAxes, ha='left', va='bottom', fontsize=9,
                color='0.2')
        if j == 0:
            ax.set_ylabel('Depth [km]', fontsize=10)
            ax.legend(frameon=False, fontsize=8.5, loc='upper left')
        else:
            ax.tick_params(labelleft=False)
        axp[1].set_ylim(PROFILE_ZMAX_KM, 0)

    # ---- the table --------------------------------------------------------
    print()
    for k in ('STD', 'WAL'):
        t = d[f'{k}_t']; nd, rot, tilt, V = S[k]
        dip = np.degrees(d[f'{k}_dip'])
        sh = tilt - nd
        amax, amed = np.abs(nd).max(), np.median(np.abs(nd))
        zc = ZC_KM * 1e3; wz = cc['z'] <= zc
        # the shared cell rule, 0 -> z_c
        gT = -cpc.integrate_to_zc(cc[f'{k}_szz_T'], cc['z'], axis=1)
        gI = -cpc.integrate_to_zc(cc[f'{k}_szz_I'], cc['z'], axis=1)
        TP = np.median(np.abs(gI - gT)) / 1e12
        agree = int((np.sign(tilt) == np.sign(nd)).sum())
        i = pick[k]
        print(f'{k}: dip {dip.min():.1f}-{dip.max():.1f} deg (median '
              f'{np.median(dip):.1f}) | N_D(x_T) {nd.min():+.2f}..{nd.max():+.2f}'
              f', median |N_D| {amed:.2f}, max {amax:.2f}')
        print(f'     plane shift median {np.median(sh):+.3f} TN/m = '
              f'{100*np.median(sh)/amax:.0f} % of max |N_D|, '
              f'{100*np.median(sh)/amed:.0f} % of median |N_D|, '
              f'{100*np.median(sh)/TP:.0f} % of the trench pull ({TP:.2f})')
        print(f'     sign of N_D(x_T) agrees on {agree}/{len(nd)} snapshots | '
              f'unambiguous epoch t = {t[i]:.0f} Myr: {nd[i]:+.3f} vertical, '
              f'{tilt[i]:+.3f} slab-normal ({100*(tilt[i]-nd[i])/amax:+.0f} % '
              f'of max |N_D|)')
        rows += [
            (k, 'dip_median_deg', f'{np.median(dip):.2f}'),
            (k, 'dip_min_deg', f'{dip.min():.2f}'),
            (k, 'dip_max_deg', f'{dip.max():.2f}'),
            (k, 'nd_vertical_median_TNm', f'{np.median(nd):.3f}'),
            (k, 'nd_vertical_min_TNm', f'{nd.min():.3f}'),
            (k, 'nd_vertical_max_TNm', f'{nd.max():.3f}'),
            (k, 'abs_nd_median_TNm', f'{amed:.3f}'),
            (k, 'abs_nd_max_TNm', f'{amax:.3f}'),
            (k, 'V_median_TNm', f'{np.median(V):.3f}'),
            (k, 'trench_pull_median_TNm', f'{TP:.3f}'),
            (k, 'plane_shift_median_TNm', f'{np.median(sh):.3f}'),
            (k, 'plane_shift_pct_of_max_abs_nd', f'{100*np.median(sh)/amax:.1f}'),
            (k, 'plane_shift_pct_of_median_abs_nd', f'{100*np.median(sh)/amed:.1f}'),
            (k, 'plane_shift_pct_of_trench_pull', f'{100*np.median(sh)/TP:.1f}'),
            (k, 'plane_shift_min_TNm', f'{sh.min():.3f}'),
            (k, 'plane_shift_max_TNm', f'{sh.max():.3f}'),
            (k, 'rotation_only_shift_median_TNm', f'{np.median(rot-nd):.3f}'),
            (k, 'shift_positive_at_every_snapshot', str(bool((sh > 0).all()))),
            (k, 'V_positive_at_every_snapshot', str(bool((V > 0).all()))),
            (k, 'tension_like_pct_vertical', f'{100*np.mean(nd>0):.0f}'),
            (k, 'tension_like_pct_slabnormal', f'{100*np.mean(tilt>0):.0f}'),
            (k, 'sign_agreement_snapshots', f'{agree}/{len(nd)}'),
            (k, 'unambiguous_epoch_Myr', f'{t[i]:.1f}'),
            (k, 'unambiguous_nd_vertical_TNm', f'{nd[i]:.3f}'),
            (k, 'unambiguous_nd_slabnormal_TNm', f'{tilt[i]:.3f}'),
        ]

    out = os.path.join(ROOT, 'figures', 'fig_stress_state.png')
    depth_ticks(fig)   # 20 km depth ticks (FIGURE_STYLE)
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print('\nwritten:', out)
    tab = write_table('stress_state', rows[0], rows[1:],
                      script='fig_stress_state.py',
                      figure='fig_stress_state.png', models=('STD', 'WAL'),
                      meta={'reference_Myr': 40.0, 'zc_km': ZC_KM,
                            'window_half_width_km': 5.0,
                            'dip': '800 C isotherm fitted over +/-10 km at x_T',
                            'planes': 'vertical; slab-normal to the same depth',
                            'scaling': 'shift quoted against run-scale |N_D| '
                                       'and the trench pull, never against the '
                                       'instantaneous resultant (Dan 2026-10-09)',
                            'source': 'dip method ported from '
                                      'further_analysis/fluidity_slab_normal_FD.ipynb'})
    print('written:', tab)


if __name__ == '__main__':
    main()
