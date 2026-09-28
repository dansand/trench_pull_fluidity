"""fig_mechanical_thickness — how the plate's mechanical thickness is
defined, how the definitions evolve, and whether the trench pull's force
length scale follows h/2.

Writes figures/fig_mechanical_thickness.png and tables/mechanical_thickness.csv
from the committed column-profile cache. The FLEXURE-based definitions
(neutral plane, yield envelope, thermal) are evaluated at the column of
MAXIMUM BENDING MOMENT, which sits ~20-25 km seaward of the trench; the
deficit-based ones (triangle, TP-90 %, moment arm) are referenced to the
trench, where the force balance is taken.

The definitions (Dan's list, 2026-09-16, plus two of our own):

  thermal        the 900 C isotherm — the isotherm the two strength-based
                 definitions calibrate to (T_MECH_C below; the suite
                 convention of 950 C is ~5 km deeper and fails the h/2 test)
  2 x h_np       twice the neutral-plane depth, the neutral plane taken as
                 the extremum of the cumulative fibre stress
                 C(z) = int (sigma_xx - sigma_zz) dz  (a symmetric plate
                 bends about its mid-depth, so h = 2 h_np)
  yield 10%      the depth at which the fibre-stress envelope falls to 10 %
                 of its peak — Dan's preferred threshold, a scale-free
                 version of McNutt & Menard's fixed 50 MPa
  yield 50 MPa   the McNutt & Menard threshold, for comparison
  TP 90%         the depth at which the cumulative trench pull reaches 90 %
                 of its final value: the depth over which the topographic
                 load is actually supported
  triangle       2 x (trench pull) / (trench pressure deficit): the
                 equivalent thickness of a linearly decaying deficit

Also computed: the EFFECTIVE MOMENT ARM of the trench pull — the centroid
of the trench pressure deficit, arm = int p_T z dz / int p_T dz — and its
ratio to each thickness.

THREE PANELS SINCE 2026-09-26 (Dan). The former middle panel — trench pull
against mechanical thickness — is dropped: it carried the same thickness
ambiguity as the scaling test while testing nothing, and the trench-pull
medians it displayed remain in the table. In its place the scaling test is
SPLIT BY MODEL, one square per run. That is the panel worth separating —
37 points per run, each with a horizontal span, overlapped badly when
combined — while the time series stays combined, where navy and magenta
read cleanly apart. The two squares carry identical limits, aspect and
reference lines so they compare directly.

⚠ THE SIGMA_ZZ ZERO CROSSING IS NOT A MECHANICAL-THICKNESS ESTIMATE
(Dan's ruling, 2026-09-26): it is too sensitive to temporal fluctuation —
the crossing moves 3.7 km (STD) / 5.4 km (WAL) per MPa of shift in a
profile whose whole deep offset is only 9.6 / 4.8 MPa (fig_plate_thickness
quantifies this). The retained set is the neutral plane and the 10 % yield
cap, with the thermal curve showing that the cap essentially tracks
temperature.

DRAFT CAPTION. (a) Proxies for the plate's mechanical thickness at the
column of maximum bending moment through the runs, for STD (navy) and WAL
(magenta): twice the neutral-plane depth, the truncated yield-stress
envelope (the depth at which the fibre stress falls to 10 per cent of its
peak), and the 900 C isotherm. The yield-envelope and thermal proxies
track one another through both runs while the neutral-plane estimate stays
nearly constant. (b, c) The trench pull force h/2 scaling test, one panel
per model: the force length scale L = trench pull / surface deficit against h/2,
one point per snapshot, the horizontal span covering the spread of the
three thickness estimates. Both panels use identical axes; the solid line
is 1:1 and the dotted line the companion manuscript's 0.55 h scaling. Run
medians of L/h are 0.54 (STD) and 0.61 (WAL) against the neutral plane,
0.49 and 0.50 against the yield envelope, and 0.52 in both against the
900 C isotherm. L falls inside the span of the three estimates on 73 per
cent of STD snapshots and 49 per cent of WAL snapshots.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import column_profiles_cache as cpc
from tables_io import write_table

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = {'STD': '#002147', 'WAL': '#E5007D'}
C_RULE = '#BFC3D1'
T_MECH_C = 900.0           # the isotherm matching the strength-based definitions
                           # (measured 898 STD / 892 WAL -> 900, Dan 2026-09-16).
                           # The suite convention is 950 C (conventions §5.2), which
                           # this analysis says is ~5 km too deep: it fails the h/2
                           # scaling test at L/h = 0.46-0.47 where 900 C gives 0.51.
YIELD_FRAC = 0.10          # Dan's scale-free threshold
YIELD_ABS_MPA = 50.0       # McNutt & Menard
TP_FRAC = 0.90             # trench-pull equilibration level

def thicknesses(d, key):
    """All definitions, per snapshot, evaluated at the trench column."""
    c = cpc.derive(d, key)
    p = cpc.partition(d, key)
    z = c['z']
    sm = lambda a: gaussian_filter1d(a, 2)
    # Flexure-based definitions are evaluated at the MAXIMUM-BENDING-MOMENT
    # column (Dan, 2026-09-16), which sits ~20-25 km seaward of the trench;
    # the deficit-based ones (triangle, TP-90 %, arm) stay at the trench,
    # where the force balance is referenced.
    fib = d[f'{key}_sxx_M'] - d[f'{key}_szz_M']        # fibre stress, Pa
    temp = d[f'{key}_temp_M'] - 273.0
    out = {k: [] for k in ('thermal', 'np2', 'yield10', 'yield50', 'tp90',
                           'triangle', 'arm', 'trench_pull', 'h_strength',
                           'T_strength')}
    for i in range(len(c['t'])):
        f = sm(fib[i])
        # thermal
        j = np.where(temp[i] >= T_MECH_C)[0]
        out['thermal'].append(z[j[0]] if len(j) else np.nan)
        # neutral plane: extremum of the cumulative fibre stress (5-60 km window)
        cfib = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(z))])
        w = (z >= 5e3) & (z <= 60e3)
        h_np = z[w][int(np.argmax(np.abs(cfib[w])))]
        out['np2'].append(2 * h_np)
        # yield-envelope thresholds, evaluated FROM THE BOTTOM UP (Dan,
        # 2026-09-16): the base is the DEEPEST level at which the envelope
        # still exceeds the threshold. Searching downward from the neutral
        # plane instead lands in the zero crossing of the envelope itself
        # — a ~1 km window on this grid — and returns h_np, not the base.
        peak = np.abs(f[(z > 2e3) & (z < 80e3)]).max()
        deep = z < 150e3
        k10 = np.where(deep & (np.abs(f) >= YIELD_FRAC * peak))[0]
        k50 = np.where(deep & (np.abs(f) >= YIELD_ABS_MPA * 1e6))[0]
        out['yield10'].append(z[k10[-1]] if len(k10) else np.nan)
        out['yield50'].append(z[k50[-1]] if len(k50) else np.nan)
        # trench-pull equilibration and the moment arm
        pT = sm(c['p_T'][i])                    # pressure register (deficit negative)
        dfc = -pT                                # deficit, positive
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (dfc[1:] + dfc[:-1]) * np.diff(z))])
        zc = z <= 120e3
        total = cum[zc][-1]
        k = np.where(cum >= TP_FRAC * total)[0]
        out['tp90'].append(z[k[0]] if len(k) else np.nan)
        out['trench_pull'].append(total)
        dP_T = np.abs(dfc[z <= 75e3]).max()
        out['triangle'].append(2 * total / dP_T)
        out['arm'].append(np.trapz(dfc[zc] * z[zc], z[zc]) / total)
        # the isotherm that matches the mean of the three strength-based
        # definitions (Dan, 2026-09-16): 2 h_np, yield-10 %, triangle
        # the strength-based mean: the two definitions derived from the
        # stress envelope (the thermal one is what it calibrates, the
        # triangle is force-derived — neither belongs here)
        h_bar = np.nanmean([out['np2'][-1], out['yield10'][-1]])
        out['h_strength'].append(h_bar)
        out['T_strength'].append(np.interp(h_bar, z, temp[i]))
    out = {k: np.array(v) for k, v in out.items()}
    out['t'] = c['t']
    return out

def main():
    if not hasattr(np, 'trapz'):
        np.trapz = np.trapezoid
    d = cpc.load()
    if f'STD_sxx_T' not in d.files:
        raise SystemExit('cache lacks sigma_xx — rebuild column_profiles_cache.py')
    # THREE PANELS since 2026-09-26 (Dan): the definitions through time across
    # the top, and the h/2 scaling test split into ONE SQUARE PER MODEL below.
    # The scatter is where the two runs overlap worst -- 37 points with
    # horizontal spans each -- so that is the panel worth separating; the time
    # series stays combined, where navy and magenta read cleanly apart.
    # The former trench-pull-against-thickness panel stays dropped; its
    # numbers are in the table (trench_pull_median_TNm and the h_* medians).
    fig = plt.figure(figsize=(12.5, 10.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.30, wspace=0.18)
    ax_t = fig.add_subplot(gs[0, :])
    ax_sc = {'STD': fig.add_subplot(gs[1:, 0]),
             'WAL': fig.add_subplot(gs[1:, 1])}
    # Trimmed to the four informative definitions (Dan, 2026-09-16);
    # yield-50 MPa and the 90 %-equilibration depth remain in the table.
    # Three estimates (Dan, 2026-09-16): the neutral plane, the truncated
    # yield-stress envelope, and the 900 C isotherm. The triangle equivalent
    # is NOT a fourth estimate — it is 2 L, i.e. the force-derived length
    # being tested in panel (c), so including it would be circular. The
    # 50 MPa envelope stays in the table only.
    styles = [('np2', '-o', r'$2\,h_{np}$  (neutral plane)'),
              ('yield10', '--', 'truncated yield-stress envelope (10 % of peak)'),
              ('thermal', '-', 'thermal, 900 $^\\circ$C')]
    rows = [('model', 'quantity', 'value')]
    for key in ('STD', 'WAL'):
        col = C[key]
        r = thicknesses(d, key)
        for name, ls, lab in styles:
            kw = dict(color=col, lw=1.6)
            if ls == '-o':
                ax_t.plot(r['t'], r[name] / 1e3, ls, ms=4, markeredgecolor='white',
                          markeredgewidth=0.5, **kw)
            else:
                ax_t.plot(r['t'], r[name] / 1e3, ls=ls, **kw)
            rows += [(key, f'h_{name}_median_km', f'{np.nanmedian(r[name])/1e3:.1f}'),
                     (key, f'h_{name}_q1_km', f'{np.nanpercentile(r[name], 25)/1e3:.1f}'),
                     (key, f'h_{name}_q3_km', f'{np.nanpercentile(r[name], 75)/1e3:.1f}')]

        # (c) THE h/2 SCALING TEST, one point per snapshot (Dan, 2026-09-16):
        #   y  L = trench pull / surface pressure deficit — a single value
        #   x  h/2, with a horizontal SPAN covering the spread of the
        #      independent thickness estimates (2 h_np, yield-10 %,
        #      yield-50 MPa, thermal 900 C). The triangle definition is
        #      EXCLUDED from the span: L = triangle/2 identically, so
        #      including it would make the test circular.
        L = r['triangle'] / 2
        ests = np.vstack([r['np2'], r['yield10'], r['thermal']]) / 2e3
        lo, hi = np.nanmin(ests, axis=0), np.nanmax(ests, axis=0)
        mid = np.nanmedian(ests, axis=0)
        ax_sc[key].errorbar(mid, L / 1e3, xerr=[mid - lo, hi - mid], fmt='o',
                            color=col, ms=5, lw=0, elinewidth=1.0, capsize=2,
                            alpha=0.85, markeredgecolor='white',
                            markeredgewidth=0.5)   # model named in the title
        print(f'   h/2 test: L/(h/2) using the estimate median = '
              f'{np.nanmedian(L / 1e3 / mid):.2f}; '
              f'L inside the estimate span in {np.mean((L/1e3 >= lo) & (L/1e3 <= hi))*100:.0f}% of snapshots')
        ratio = r['arm'] / r['np2']
        rows += [(key, 'strength_mean_thickness_km', f'{np.nanmedian(r["h_strength"])/1e3:.1f}'),
                 (key, 'matching_isotherm_C', f'{np.nanmedian(r["T_strength"]):.0f}'),
                 (key, 'h_yield50_median_km', f'{np.nanmedian(r["yield50"])/1e3:.1f}'),
                 (key, 'h_tp90_median_km', f'{np.nanmedian(r["tp90"])/1e3:.1f}'),
                 (key, 'moment_arm_median_km', f'{np.nanmedian(r["arm"])/1e3:.1f}'),
                 (key, 'moment_arm_over_h_median', f'{np.nanmedian(ratio):.3f}'),
                 (key, 'L_scaling_km_median', f'{np.nanmedian(r["triangle"]/2)/1e3:.2f}'),
                 (key, 'L_over_h_np2', f'{np.nanmedian(r["triangle"]/2/r["np2"]):.3f}'),
                 (key, 'L_over_h_yield10', f'{np.nanmedian(r["triangle"]/2/r["yield10"]):.3f}'),
                 (key, 'L_over_h_thermal900', f'{np.nanmedian(r["triangle"]/2/r["thermal"]):.3f}'),
                 (key, 'L_over_h_strengthmean', f'{np.nanmedian(r["triangle"]/2/r["h_strength"]):.3f}'),
                 (key, 'trench_pull_median_TNm', f'{np.nanmedian(r["trench_pull"])/1e12:.3f}')]
        print(f'=== {key} === (run medians, km)')
        for name, _, lab in styles:
            print(f'   {lab:34s} {np.nanmedian(r[name])/1e3:5.1f}  '
                  f'(IQR {np.nanpercentile(r[name],25)/1e3:.0f}–{np.nanpercentile(r[name],75)/1e3:.0f})')
        Lk = r['triangle'] / 2
        print(f'   L (trench pull / surface deficit)  {np.nanmedian(Lk)/1e3:5.1f} km')
        print(f'   h/2 scaling test  L/h: 2h_np {np.nanmedian(Lk/r["np2"]):.3f}, '
              f'yield10 {np.nanmedian(Lk/r["yield10"]):.3f}, '
              f'thermal900 {np.nanmedian(Lk/r["thermal"]):.3f}, '
              f'strength-mean {np.nanmedian(Lk/r["h_strength"]):.3f}')
        print(f'   centroid of the deficit (a different quantity) '
              f'{np.nanmedian(r["arm"])/1e3:.1f} km')

    ax_t.set_ylabel('Thickness [km]', fontsize=11)
    ax_t.set_xlabel('Model time [Myr]', fontsize=11)
    ax_t.set_title('(a) proxies for the mechanical thickness at the '
                   'max-moment column (navy STD, magenta WAL)', fontsize=10.5)
    ax_t.grid(alpha=0.25, color=C_RULE, lw=0.6)
    # Tightened 2026-09-26: the 20-110 km range dated from the
    # six-definition version and left the curves in the lower third.
    ax_t.set_ylim(40, 90)
    for _, ls, lab in styles:
        if ls == '-o':
            ax_t.plot([], [], '-o', color='0.35', lw=1.6, ms=4, label=lab)
        else:
            ax_t.plot([], [], color='0.35', lw=1.6, ls=ls, label=lab)
    ax_t.legend(frameon=False, fontsize=8.5, ncol=3, loc='upper left')
    # The two scaling-test squares carry IDENTICAL limits, aspect and
    # reference lines, so the eye compares them directly.
    lim = np.array([20, 45])
    for lett, key in (('b', 'STD'), ('c', 'WAL')):
        ax = ax_sc[key]
        ax.plot(lim, lim, 'k-', lw=1.2, label='1:1')
        ax.plot(lim, 1.1 * lim, 'k:', lw=0.9, label=r'$0.55\,h$')
        ax.set_xlim(*lim); ax.set_ylim(*lim)
        ax.set_box_aspect(1)
        ax.set_xlabel('$h/2$ [km]  (span: spread of estimates)', fontsize=10.5)
        ax.set_title(f'({lett}) the trench pull force $h/2$ scaling test — {key}',
                     fontsize=10.5)
        ax.grid(alpha=0.25, color=C_RULE, lw=0.6)
        ax.legend(frameon=False, fontsize=9, loc='upper left')
    ax_sc['STD'].set_ylabel('$L$ = trench pull / deficit [km]', fontsize=10.5)
    ax_sc['WAL'].tick_params(labelleft=False)
    out = os.path.join(ROOT, 'figures', 'fig_mechanical_thickness.png')
    fig.savefig(out, bbox_inches='tight', dpi=220)
    print('written:', out)
    tab = write_table('mechanical_thickness', rows[0], rows[1:],
                      script='fig_mechanical_thickness.py', figure='fig_mechanical_thickness.png',
                      models=('STD', 'WAL'), meta={'T_mech_C': T_MECH_C, 'yield_frac': YIELD_FRAC, 'yield_abs_MPa': YIELD_ABS_MPA, 'column': 'maximum bending moment'})
    print('written:', tab)

if __name__ == '__main__':
    main()
