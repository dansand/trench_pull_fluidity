"""fig_mechanical_thickness — six ways to define the plate's mechanical
thickness, how they evolve, and what trench pull does as a function of it.

Writes figures/fig_mechanical_thickness.png and tables/mechanical_thickness.csv
from the committed column-profile cache. The FLEXURE-based definitions
(neutral plane, yield envelope, thermal) are evaluated at the column of
MAXIMUM BENDING MOMENT, which sits ~20-25 km seaward of the trench; the
deficit-based ones (triangle, TP-90 %, moment arm) are referenced to the
trench, where the force balance is taken.

The definitions (Dan's list, 2026-09-16, plus two of our own):

  thermal        the 950 C isotherm — the suite's declared mechanical base
                 (conventions §5.2), the thermal reference point
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

DRAFT CAPTION. (a) Six definitions of the plate's mechanical thickness at
the trench column through the runs, for STD (navy) and WAL (magenta).
(b) Trench pull against mechanical thickness (2 x neutral-plane depth),
every snapshot. (c) The effective moment arm of the trench pull — the
centroid of the pressure deficit — against the same thickness; the dashed
line is the arm expected for a linearly decaying deficit (h/3).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import column_profiles_cache as cpc

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
        h_bar = np.nanmean([out['np2'][-1], out['yield10'][-1], out['triangle'][-1]])
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
    # 3-row grid: the time-series panel spans two columns and ONE row, each
    # lower panel takes one column and TWO rows, so the squares fill their
    # column width instead of being cramped by the row height (Dan, 2026-09-16)
    fig = plt.figure(figsize=(12.5, 10.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.30, wspace=0.22)
    ax_t = fig.add_subplot(gs[0, :])
    ax_p = fig.add_subplot(gs[1:, 0])
    ax_a = fig.add_subplot(gs[1:, 1])
    # Trimmed to the four informative definitions (Dan, 2026-09-16);
    # yield-50 MPa and the 90 %-equilibration depth remain in the table.
    styles = [('np2', '-o', r'$2\,h_{np}$  (lower bound, nearly constant)'),
              ('yield10', '--', 'yield envelope, 10 % of peak'),
              ('triangle', (0, (3, 1, 1, 1)), 'triangle equivalent (GPE)'),
              ('thermal', '-', 'thermal, 900 $^\\circ$C (upper bound, grows)')]
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
        ests = np.vstack([r['np2'], r['yield10'], r['yield50'], r['thermal']]) / 2e3
        lo, hi = np.nanmin(ests, axis=0), np.nanmax(ests, axis=0)
        mid = np.nanmedian(ests, axis=0)
        # (b) trench pull against thickness, SAME span as (c) — the
        # thickness ambiguity is shown in both panels (Dan, 2026-09-16)
        ax_p.errorbar(2 * mid, r['trench_pull'] / 1e12,
                      xerr=[2 * (mid - lo), 2 * (hi - mid)], fmt='o', color=col,
                      ms=5, lw=0, elinewidth=1.0, capsize=2, alpha=0.85,
                      markeredgecolor='white', markeredgewidth=0.5, label=key)
        ax_a.errorbar(mid, L / 1e3, xerr=[mid - lo, hi - mid], fmt='o', color=col,
                      ms=5, lw=0, elinewidth=1.0, capsize=2, alpha=0.85,
                      markeredgecolor='white', markeredgewidth=0.5, label=key)
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
    ax_t.set_title('(a) definitions of the mechanical thickness at the max-moment '
                   'column\n(navy STD, magenta WAL); the strength-based mean '
                   'matches the 900 $^\\circ$C isotherm', fontsize=10.5)
    ax_t.grid(alpha=0.25, color=C_RULE, lw=0.6)
    ax_t.set_ylim(20, 110)
    for _, ls, lab in styles:
        if ls == '-o':
            ax_t.plot([], [], '-o', color='0.35', lw=1.6, ms=4, label=lab)
        else:
            ax_t.plot([], [], color='0.35', lw=1.6, ls=ls, label=lab)
    ax_t.legend(frameon=False, fontsize=8.5, ncol=3, loc='upper left')
    ax_p.set_xlabel('mechanical thickness $h$ [km]  (span: spread of estimates)',
                    fontsize=10.5)
    ax_p.set_ylabel('trench pull [TN/m]', fontsize=10.5)
    ax_p.set_xlim(36, 100)
    ax_p.set_ylim(0.8, 3.6)
    ax_p.set_box_aspect(1)
    ax_p.set_title('(b) trench pull vs mechanical thickness', fontsize=10.5)
    lim = np.array([18, 50])
    ax_a.plot(lim, lim, 'k-', lw=1.2, label='1:1')
    ax_a.plot(lim, 1.1 * lim, 'k:', lw=0.9, label=r'$0.55\,h$')
    ax_a.set_xlim(*lim); ax_a.set_ylim(*lim)
    ax_a.set_box_aspect(1)
    ax_a.set_xlabel('$h/2$ [km]  (span: spread of estimates)', fontsize=10.5)
    ax_a.set_ylabel('$L$ = trench pull / deficit [km]', fontsize=10.5)
    ax_a.set_title('(c) the $h/2$ scaling test', fontsize=10.5)
    for ax in (ax_p, ax_a):
        ax.grid(alpha=0.25, color=C_RULE, lw=0.6)
        ax.legend(frameon=False, fontsize=9)
    out = os.path.join(ROOT, 'figures', 'fig_mechanical_thickness.png')
    fig.savefig(out, bbox_inches='tight', dpi=220)
    print('written:', out)
    os.makedirs(os.path.join(ROOT, 'tables'), exist_ok=True)
    tab = os.path.join(ROOT, 'tables', 'mechanical_thickness.csv')
    with open(tab, 'w') as f:
        f.write('\n'.join(','.join(x) for x in rows) + '\n')
    print('written:', tab)

if __name__ == '__main__':
    main()
