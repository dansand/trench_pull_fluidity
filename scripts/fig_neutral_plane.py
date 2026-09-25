"""fig_neutral_plane — why the neutral-plane thickness barely grows.

Writes figures/fig_neutral_plane.png and tables/neutral_plane.csv from the
committed column-profile cache.

Dan's question, 2026-09-26, on seeing fig_plate_thickness: the
neutral-plane estimate 2 h_np is strikingly flat through both runs while
the thermal and yield estimates climb steadily. His hypothesis: the plate
thickens, which would deepen the neutral plane, but it also becomes
increasingly COMPRESSIONAL, which lifts it -- and the two nearly cancel.

That is exactly what happens, and it is beam theory rather than a
coincidence. A beam carrying an axial load N as well as a moment M has its
neutral axis displaced from mid-depth by e = -N h^2 / (12 M), so

    2 h_np - h  =  -N h^2 / (6 M)

with N the normal-stress-difference resultant and M the moment, both taken
at the column of MAXIMUM BENDING MOMENT where h_np itself is evaluated.

THE TREND DECOMPOSITION, per 10 Myr:

                                      STD        WAL
    thickening alone would give     +3.54 km    +3.67 km
    the N/M term contributes        -2.82       -1.08
    observed 2 h_np trend           +0.47       +1.10

So in STD the compressional term cancels 80 % of the thickening. N_D at
the maximum-moment column falls at -0.48 TN/m per 10 Myr in STD and -0.43
in WAL; its medians are -0.39 and -1.48 TN/m, so WAL is the more
compressional run throughout.

THE TEST. Panel (c) plots the measured offset against the beam
prediction, one point per snapshot. The fitted slopes are 1.14 (STD) and
1.21 (WAL) against the predicted 1.0, and splitting by offset magnitude
gives 1.13 / 1.14 in STD and 1.14 / 1.00 in WAL -- so the relation holds
across the whole range of offsets in both runs. The ~15 % excess is
consistent and worth stating as such rather than rounded away.

⚠ THE SLOPE MUST BE TAKEN FROM THE QUANTITY PANEL (c) PLOTS -- the full
prediction with h at its per-snapshot value. An earlier draft of this
docstring quoted slopes from an h-FROZEN variant (1.01 and 1.85) and read
WAL's as a pooling artefact; that variant is a diagnostic, not the
prediction, and its slopes are not comparable to unity. Corrected
2026-09-26.

⚠ The levels correlation IS inflated by construction, since both the
measured offset and the prediction carry h^2. Freezing h so that only
N/M varies is the check for that, and the detrended correlation survives
it: +0.63 (STD) and +0.80 (WAL).

WHAT IT MEANS FOR THE PAPER. 2 h_np IS NOT A THICKNESS when the plate
carries significant axial load; it is a thickness minus a stress-state
term, and in STD the two nearly cancel. Anywhere the paper uses the
neutral plane as a mechanical thickness -- including the h/2 dipole
scaling in fig_mechanical_thickness -- this correction applies.

  1, 2  the three quantities through time, one panel per model: the
        thermal thickness, the raw neutral-plane estimate, and the
        neutral-plane estimate with the axial term added back, which
        should recover the thermal curve if the relation holds.
  3     the test: measured offset against beam prediction, with the
        one-to-one line.

DRAFT CAPTION. (a, b) The thermal thickness of the plate (900 C isotherm),
twice the neutral-plane depth, and twice the neutral-plane depth corrected
for the axial load by adding N h^2/(6 M), for STD and WAL. All three are
evaluated at the column of maximum bending moment, with N the
normal-stress-difference resultant and M the bending moment there. The raw
neutral-plane estimate grows by 0.5 and 1.1 km per 10 Myr while the
thermal estimate grows by 3.5 and 3.7; the corrected estimate recovers the
thermal trend. (c) The measured offset of the neutral-plane estimate from
the thermal thickness against the offset predicted by beam theory for a
plate under combined axial load and bending, one point per snapshot, with
the one-to-one line.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import column_profiles_cache as cpc
from fig_mechanical_thickness import thicknesses
from fig_budget_time import T_MIN_MYR
from tables_io import write_table

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C_RULE = '#BFC3D1'
C_MODEL = {'STD': '#002147', 'WAL': '#E5007D'}
C_TH = '#0072B2'
C_NP = '#009E73'
C_CORR = '#7B3294'
ZC = 75e3


def parts(d, key):
    """Thicknesses, the axial term, and the beam offset, per snapshot."""
    if not hasattr(np, 'trapezoid'):
        np.trapezoid = np.trapz
    c = cpc.derive(d, key)
    z, t = c['z'], c['t']
    r = thicknesses(d, key)
    zc = z <= ZC
    # N_D and M at the MAXIMUM-MOMENT column -- the same column at which
    # h_np is evaluated, which is what makes the comparison legitimate.
    nd = np.trapezoid((d[f'{key}_sxx_M'] - d[f'{key}_szz_M'])[:, zc],
                      z[zc], axis=1)
    mom = d[f'{key}_M_max']
    h, np2 = r['thermal'], r['np2']
    pred = -nd * h ** 2 / (6.0 * mom)          # beam offset, 2h_np - h
    return dict(t=t, m=t >= T_MIN_MYR, h=h, np2=np2, nd=nd, mom=mom,
                pred=pred, res=np2 - h, corr=np2 - pred)


def main():
    d = cpc.load()
    fig = plt.figure(figsize=(11.0, 7.6), layout='constrained')
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.85])
    axes = [fig.add_subplot(gs[0, c]) for c in range(2)]
    axs = fig.add_subplot(gs[1, :])
    rows = [('model', 'quantity', 'value')]

    for col, key in enumerate(('STD', 'WAL')):
        p = parts(d, key)
        t, m = p['t'], p['m']
        ax = axes[col]
        ax.plot(t[m], p['h'][m] / 1e3, '-', color=C_TH, lw=2.0,
                label='thermal, 900 $^\\circ$C')
        ax.plot(t[m], p['np2'][m] / 1e3, '--', color=C_NP, lw=2.0,
                label=r'$2\,h_{np}$ (raw)')
        ax.plot(t[m], p['corr'][m] / 1e3, ':', color=C_CORR, lw=2.2,
                label=r'$2\,h_{np}$ + axial term')
        ax.set_title(key, fontsize=11.5)
        ax.set_xlabel('Model time [Myr]', fontsize=10.5)
        ax.grid(alpha=0.25, color=C_RULE, lw=0.6)
        if col == 0:
            ax.set_ylabel('Thickness [km]', fontsize=11)
            ax.legend(frameon=False, fontsize=8.5, loc='upper left')

        axs.plot(p['pred'][m] / 1e3, p['res'][m] / 1e3, 'o', ms=5,
                 color=C_MODEL[key], alpha=0.8, label=key)

        sl = lambda a: np.polyfit(t[m], a[m], 1)[0] * 10 / 1e3
        # h FROZEN so that only N/M varies -- the levels correlation is
        # inflated because both sides carry h^2.
        hm = np.median(p['h'][m])
        pf = (-p['nd'] * hm ** 2 / (6 * p['mom']))[m]
        rs = p['res'][m]
        ld = lambda a: a - np.polyval(np.polyfit(t[m], a, 1), t[m])
        # THE SLOPE IS TAKEN FROM THE QUANTITY PLOTTED IN PANEL (c) -- the
        # full prediction, with h at its per-snapshot value. The h-frozen
        # version below is only a check that the agreement is not
        # manufactured by h^2 appearing on both sides; it is NOT the
        # prediction, and its slope should not be quoted as one.
        pv = p['pred'][m]
        rel = np.abs(rs / p['h'][m])
        lo_, hi_ = rel < np.median(rel), rel >= np.median(rel)
        s_pool = np.polyfit(pv, rs, 1)[0]
        s_small = np.polyfit(pv[lo_], rs[lo_], 1)[0]
        s_large = np.polyfit(pv[hi_], rs[hi_], 1)[0]
        print(f'{key}: trends/10 Myr  thermal {sl(p["h"]):+.2f}, 2h_np raw '
              f'{sl(p["np2"]):+.2f}, corrected {sl(p["corr"]):+.2f} km;  '
              f'axial term {np.polyfit(t[m], pf, 1)[0]*10/1e3:+.2f};  '
              f'beam slope pooled {s_pool:.2f} (small {s_small:.2f}, large '
              f'{s_large:.2f});  h-frozen check: r(detrended) '
              f'{np.corrcoef(ld(rs), ld(pf))[0, 1]:+.2f}')
        rows += [(key, 'trend_thermal_km_per_10Myr', f'{sl(p["h"]):.2f}'),
                 (key, 'trend_2hnp_raw_km_per_10Myr', f'{sl(p["np2"]):.2f}'),
                 (key, 'trend_2hnp_corrected_km_per_10Myr', f'{sl(p["corr"]):.2f}'),
                 (key, 'trend_axial_term_km_per_10Myr',
                  f'{np.polyfit(t[m], pf, 1)[0]*10/1e3:.2f}'),
                 (key, 'nd_at_max_moment_median_TNm',
                  f'{np.median(p["nd"][m])/1e12:.3f}'),
                 (key, 'nd_at_max_moment_trend_TNm_per_10Myr',
                  f'{np.polyfit(t[m], p["nd"][m], 1)[0]*10/1e12:.3f}'),
                 (key, 'beam_slope_pooled', f'{s_pool:.3f}'),
                 (key, 'beam_slope_small_offset', f'{s_small:.3f}'),
                 (key, 'beam_slope_large_offset', f'{s_large:.3f}'),
                 (key, 'beam_r_detrended_h_frozen',
                  f'{np.corrcoef(ld(rs), ld(pf))[0, 1]:.3f}')]

    lo = min(min(a.get_ylim()) for a in axes)
    hi = max(max(a.get_ylim()) for a in axes)
    for a in axes:
        a.set_ylim(lo, hi)
    lim = np.array(axs.get_xlim())
    axs.plot(lim, lim, '-', color='0.4', lw=1.2, label='one-to-one')
    axs.set_xlim(*lim)
    axs.set_xlabel(r'Beam prediction, $-N h^{2}/(6M)$ [km]', fontsize=11)
    axs.set_ylabel('Measured offset,\n' r'$2\,h_{np} - h$ [km]', fontsize=10.5)
    axs.grid(alpha=0.25, color=C_RULE, lw=0.6)
    axs.legend(frameon=False, fontsize=9.5, loc='upper left')
    fig.suptitle('Why the neutral-plane thickness barely grows',
                 fontsize=10.5, color='0.35')
    out = os.path.join(ROOT, 'figures', 'fig_neutral_plane.png')
    fig.savefig(out, bbox_inches='tight', dpi=220)
    print('written:', out)
    tab = write_table('neutral_plane', rows[0], rows[1:],
                      script='fig_neutral_plane.py',
                      figure='fig_neutral_plane.png', models=('STD', 'WAL'),
                      meta={'relation': '2 h_np - h = -N h^2 / (6 M), beam under '
                                        'combined axial load and bending',
                            'column': 'maximum bending moment, for h_np, N and M',
                            'nd_depth_limit_km': ZC / 1e3, 't_min_Myr': T_MIN_MYR,
                            'slopes': 'quote the split slopes; the pooled WAL value '
                                      'is a pooling artefact'})
    print('written:', tab)


if __name__ == '__main__':
    main()
