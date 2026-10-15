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
  yield 25 MPa   the depth below which the fibre stress stays under
                 25 MPa — one physical stress level for both models
  yield 50 MPa   our translation of McNutt & Menard's criterion onto the
                 carried stress, for comparison (see YIELD_ABS_MPA)
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
ambiguity as the scaling test while testing nothing, and the full-lobe
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
quantifies this). The retained set is the neutral plane and the 25 MPa yield
cap, with the thermal curve showing that the cap essentially tracks
temperature.

DRAFT CAPTION. (a) Mechanical thickness h_m at the trench column through
the STD (navy) and WAL (magenta) runs, with the 900 C isotherm dotted.
(b) Effective arm of trench pull. The measured arm is
a_eff = |Delta GPE*| / Delta P_T, where Delta P_T is the surface pressure
deficit at the trench; the predicted arm is half the mechanical thickness
(Sandiford, 2026). Shading spans 0.5 h_m +- 10 per cent, its upper edge
coinciding with that study's fitted 0.55 h_m. Median a_eff/h_m is 0.487
(STD) and 0.504 (WAL), and 72 of the 74 snapshots fall inside the band.

WHY THE BAND AND NOT A REGRESSION. See BAND_LO/BAND_HI below: the slope of
a_eff on h_m/2 is not stable against the integration limit (0.90 at z_c,
0.98 at 120 km, 1.02 at 200 km, no plateau), while the median ratio moves
by 2 per cent across that whole range and the hit rate does not move. The
panel therefore claims the ratio and its spread, which survive the choice,
and does not quote a slope, which does not.
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
import fluidity_helpers as fh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = {'STD': '#002147', 'WAL': '#E5007D'}
C_RULE = '#BFC3D1'
T_MECH_C = 900.0           # CALIBRATED, not independent: chosen to match the
                           # strength cap at mid-run. Say so wherever it is quoted
                           # -- the two agree in median BY CONSTRUCTION. They are
                           # not redundant, though: through time they correlate at
                           # only +0.14 / +0.29 and grow at +3.50 vs +0.88 km per
                           # 10 Myr, and that divergence is the SI §S4 result.
                           # (measured 898 STD / 892 WAL -> 900, Dan 2026-09-16).
                           # The suite convention is 950 C (conventions §5.2), which
                           # this analysis says is ~5 km too deep: it fails the h/2
                           # scaling test at L/h = 0.46-0.47 where 900 C gives 0.51.
YIELD_CAP_MPA = 25.0       # THE STRENGTH CAP (Dan, 2026-10-10). Was 10 % of the
                           # per-snapshot peak, which cut STD at 26.3 MPa and WAL
                           # at 25.1 -- two different physical stress levels, for
                           # the same reason one z_c serves both models. 25 MPa is
                           # 10 % of the median peak (263/251 MPa), so the measure
                           # is unchanged: identical mid-run medians, never more
                           # than 2 km apart at any snapshot. 20 MPa was tried and
                           # rejected -- stable at mid-run but running away by
                           # ~50 km at some states, the failure mode that
                           # disqualified the sigma_zz crossing.
YIELD_ABS_MPA = 50.0       # McNutt & Menard's 50 MPa, for comparison. ⚠ THEIRS IS
                           # A YIELD-STRENGTH criterion -- "the depth at which the
                           # yield strength at geological strain rates is less than
                           # 50 MPa" -- and this applies it to the CARRIED fibre
                           # stress, which through the elastic core sits well below
                           # the envelope. It is our translation of their criterion,
                           # not their criterion, and cuts shallower.
TP_FRAC = 0.90             # trench-pull equilibration level
# THE TOLERANCE BAND ON THE HALF-THICKNESS ARM (Dan, 2026-10-12). The claim
# this panel carries is a_eff = 0.5 h_m to within +-10 %, which the band
# draws and the hit rate measures. It REPLACES the regression slope as the
# statistic: the slope is not stable against the integration depth (it runs
# 0.90 at z_c, 0.98 at 120 km and 1.02 at 200 km, with no plateau to appeal
# to), whereas the median ratio is flat to 2 % over that whole range and the
# hit rate does not move at all. Quoting a number that a defensible-looking
# change of limit can shift by 0.1 is the thing to avoid here.
# 0.55 is the COMPANION's elasto-plastic fit; 0.45 is simply its mirror
# about 0.5, not a second published bound -- the caption must say so.
BAND_LO, BAND_HI = 0.45, 0.55
# WHICH COLUMN THE FLEXURE-BASED THICKNESSES ARE READ AT.
# 'T' = THE TRENCH (default, Dan 2026-10-11): the force balance is
#       referenced to the trench, so the thickness and the force it is
#       compared against now describe the SAME column. It also tightens the
#       scaling test slightly, a_eff/(h/2) = 0.997 / 1.028 against the
#       max-moment column's 1.005 / 1.033.
# 'M' = the maximum-bending-moment column (the outer rise), the previous
#       default, kept for comparison: FLEX_COL=M python fig_...py
# ⚠ The neutral-plane CROSS-CHECK pick (first zero crossing, conventions
# §5.3) misbehaves at the trench: WAL t = 70 and 72 Myr have TWO crossings
# inside the 5-60 km window, the first at 5.5-6.5 km, so the cross-check
# reads ~6 km where the pick actually used reads 29-30. Raising the window
# floor from 5 to 8 km removes it (max discrepancy 23.2 -> 0.5 km). Not
# acted on here: the neutral plane is deprecated as a proxy in this figure
# and the pick in use is unaffected.
FLEX_COL = os.environ.get('FLEX_COL', 'T')
_COLNAME = {'M': 'max-moment column', 'T': 'trench column'}[FLEX_COL]

def thicknesses(d, key, col=None):
    """All definitions, per snapshot. `col` selects the flexure column.

    col='M'  the MAXIMUM-BENDING-MOMENT column (the outer rise), ~20-25 km
             seaward of the trench -- where the flexural signal is strongest,
             which is why the strength-based definitions were read there
             (Dan, 2026-09-16).
    col='T'  the TRENCH column -- where the force balance is actually taken,
             so the thickness and the force it is compared against describe
             the SAME column (Dan, 2026-10-11).

    The deficit-based definitions (triangle, TP-90 %, arm) are ALWAYS at the
    trench whichever is chosen: they come from the pressure deficit, which is
    referenced to the trench by construction. Only the flexure-based ones
    (np2, yield_cap, yield50, thermal) move.
    """
    col = FLEX_COL if col is None else col
    c = cpc.derive(d, key)
    p = cpc.partition(d, key)
    z = c['z']
    sm = lambda a: gaussian_filter1d(a, 2)
    fib = d[f'{key}_sxx_{col}'] - d[f'{key}_szz_{col}']   # fibre stress, Pa
    temp = d[f'{key}_temp_{col}'] - 273.0
    out = {k: [] for k in ('thermal', 'np2', 'yield_cap', 'yield50', 'tp90',
                           'triangle', 'arm', 'trench_lobe', 'trench_pull_zc',
                           'h_strength',
                           'T_strength')}
    for i in range(len(c['t'])):
        f = sm(fib[i])
        # thermal
        j = np.where(temp[i] >= T_MECH_C)[0]
        out['thermal'].append(z[j[0]] if len(j) else np.nan)
        # neutral plane: extremum of the cumulative fibre stress (5-60 km window)
        # ONE CONVENTION with column_profiles_cache, which used cell sums
        # here while this used centre-to-centre trapezoids -- two cumulative
        # fibre stresses feeding the same neutral-plane pick.
        ze, cfib = cpc.cumulative_edges(f, z)      # edges, len(z)+1
        w = (ze >= 5e3) & (ze <= 60e3)             # the 5-60 km search window
        h_np = ze[w][int(np.argmax(np.abs(cfib[w])))]
        out['np2'].append(2 * h_np)
        # yield-envelope thresholds, evaluated FROM THE BOTTOM UP (Dan,
        # 2026-09-16): the base is the DEEPEST level at which the envelope
        # still exceeds the threshold. Searching downward from the neutral
        # plane instead lands in the zero crossing of the envelope itself
        # — a ~1 km window on this grid — and returns h_np, not the base.
        peak = np.abs(f[(z > 2e3) & (z < 80e3)]).max()
        deep = z < 150e3
        k10 = np.where(deep & (np.abs(f) >= YIELD_CAP_MPA * 1e6))[0]
        k50 = np.where(deep & (np.abs(f) >= YIELD_ABS_MPA * 1e6))[0]
        out['yield_cap'].append(z[k10[-1]] if len(k10) else np.nan)
        out['yield50'].append(z[k50[-1]] if len(k50) else np.nan)
        # trench-pull equilibration and the moment arm
        pT = sm(c['p_T'][i])                    # pressure register (deficit negative)
        dfc = -pT                                # deficit, positive
        # THE 120 km LOBE IS NOT EXEMPT: it is still an integral from the
        # SURFACE over cell-centred data, so it was missing the same surface
        # half-cell. The arm's numerator and denominator below move together.
        zel, cum = cpc.cumulative_edges(dfc, z)    # edges, len(z)+1
        # TWO INTEGRALS, AND THEY ARE NOT INTERCHANGEABLE (ruling 2026-10-11,
        # on the manuscript audit's Figure 9 / Figure 10 mismatch):
        #   lobe   0 -> 120 km, the FULL trench lobe. Used for the
        #          equilibration depth tp90, where the question is "how deep
        #          before this is finished" and the full lobe is the right
        #          reference.
        #   tp_zc  0 -> z_c, THE BALANCE TERM. a_eff is defined in the
        #          manuscript as |Delta GPE*| / Delta P, and Delta GPE* is
        #          the term at z_c that every other reported force uses.
        # a_eff was built on the 120 km lobe, so Figure 9's arm and Figure
        # 10's force came from different numerators -- 2.221 against 2.179
        # TN/m in STD, 1.9 %. Dan's standing rule applies: everything the
        # paper reports uses the fixed z_c.
        lobe = cum[zel <= 120e3][-1]
        tp_zc = cum[np.argmin(np.abs(zel - cpc.ZC_KM * 1e3))]
        total = lobe                               # tp90's reference only
        k = np.where(cum >= TP_FRAC * total)[0]
        out['tp90'].append(zel[k[0]] if len(k) else np.nan)
        out['trench_lobe'].append(lobe)    # to 120 km, NOT the balance term
        out['trench_pull_zc'].append(tp_zc)  # the balance term, a_eff's numerator
        # 75 km here is a PEAK-SEARCH WINDOW, not z_c: Delta P_T is the
        # peak |Delta sigma_zz|, which sits in the upper few tens of km,
        # and the bound only keeps deep noise out of the argmax. It is
        # deliberately NOT tied to cpc.ZC_KM -- the integral above uses
        # 120 km, deeper than z_c, because the trench lobe closes by then.
        # Checked when z_c moved 75 -> 80 (2026-10-09).
        dP_T = np.abs(dfc[z <= 75e3]).max()
        # a_eff = triangle/2 = tp_zc / Delta P_T, now on the balance term
        out['triangle'].append(2 * tp_zc / dP_T)
        # the arm's numerator on the SAME 0-120 km cells as `total`
        # the CENTROID of the deficit -- a different quantity from a_eff,
        # over the full lobe in both numerator and denominator
        zcc = z <= 120e3
        dzf = float(np.diff(z)[0])
        out['arm'].append((dfc[zcc] * z[zcc]).sum() * dzf / lobe)
        # the isotherm that matches the mean of the three strength-based
        # definitions (Dan, 2026-09-16): 2 h_np, the 25 MPa cap, triangle
        # the strength-based mean: the two definitions derived from the
        # stress envelope (the thermal one is what it calibrates, the
        # triangle is force-derived — neither belongs here)
        h_bar = np.nanmean([out['np2'][-1], out['yield_cap'][-1]])
        out['h_strength'].append(h_bar)
        out['T_strength'].append(np.interp(h_bar, z, temp[i]))
    out = {k: np.array(v) for k, v in out.items()}
    out['t'] = c['t']
    return out

def effective_age(d, key, col='T'):
    """Effective thermal age [Myr] of one column, per snapshot."""
    c = cpc.derive(d, key)
    return np.array([fh.hsc_age(d[f'{key}_temp_{col}'][i], c['z'])
                     for i in range(len(c['t']))])


def age_rows(d, key, r):
    """The age rows, and the two checks that license them.

    ⚠ THE AGE IS LINEAR IN MODEL TIME BUT NOT EQUAL TO IT. The slope is
    0.72, not 1: the plate ages at one Myr per Myr while subduction keeps
    replacing the trench column with material from further up the plate,
    which was younger when the run started. So quoting "the plate is N Myr
    old at time N" is wrong by ~25 Myr by the end of these runs.
    """
    c = cpc.derive(d, key)
    t, a = c['t'], effective_age(d, key, 'T')
    p = np.polyfit(t, a, 1)
    rr = float(np.corrcoef(t, a)[0, 1])
    # OVER EVERY SNAPSHOT, not the mid-run one. The mid-run column happens
    # to be among the better-behaved, and a single-snapshot check would have
    # put 3.5 % in the table against a run median of 4.9 and a worst of 9.3.
    sp_full = np.array([fh.hsc_age_spread(d[f'{key}_temp_T'][i], c['z'])[0]
                        for i in range(len(t))])
    sp_core = np.array([fh.hsc_age_spread(d[f'{key}_temp_T'][i], c['z'],
                                          fh.AGE_ISO_CORE_C)[0]
                        for i in range(len(t))])
    a_ridge = effective_age(d, key, 'R')
    hsq = (r['yield_cap'] / 1e3) / np.sqrt(a)
    iref = cpc.reference_index(t)
    return [(key, 'age_eff_trench_first_Myr', f'{a[0]:.0f}'),
            # AT THE REFERENCE EPOCH, which is what the snapshot figures and
            # the single-state numbers describe (conventions §4b)
            (key, 'age_eff_trench_at_reference_Myr', f'{a[iref]:.0f}'),
            (key, 'reference_epoch_Myr', f'{t[iref]:.1f}'),
            (key, 'age_eff_trench_last_Myr', f'{a[-1]:.0f}'),
            (key, 'age_eff_vs_model_time_slope', f'{p[0]:.2f}'),
            (key, 'age_eff_vs_model_time_intercept_Myr', f'{p[1]:.0f}'),
            (key, 'age_eff_vs_model_time_r', f'{rr:.3f}'),
            (key, 'age_eff_isotherm_spread_core_median_pct',
             f'{100 * np.median(sp_core):.1f}'),
            (key, 'age_eff_isotherm_spread_core_max_pct',
             f'{100 * sp_core.max():.1f}'),
            (key, 'age_eff_isotherm_spread_full_median_pct',
             f'{100 * np.median(sp_full):.1f}'),
            (key, 'age_eff_isotherm_spread_full_max_pct',
             f'{100 * sp_full.max():.1f}'),
            (key, 'age_eff_isotherm_core_C', '600-1000'),
            (key, 'age_eff_isotherm_full_C', '400-1100'),
            (key, 'age_eff_ridge_column_median_Myr', f'{np.nanmedian(a_ridge):.1f}'),
            (key, 'h_m_over_sqrt_age_km_per_sqrtMyr', f'{np.nanmedian(hsq):.2f}'),
            (key, 'h_m_over_sqrt_age_range',
             f'{np.nanmin(hsq):.2f}-{np.nanmax(hsq):.2f}'),
            (key, 'age_eff_isotherm_C', f'{fh.AGE_ISO_C:.0f}'),
            (key, 'age_eff_kappa_m2_s', f'{fh.KAPPA:.1e}'),
            (key, 'age_eff_scales_as', '1/kappa')]


def _age_self_test():
    """The two checks of conventions §6.3, ASSERTED (not printed).

    They are properties of the method, so a failure means the cache, the
    archive constants or the pickers have moved -- not that the number
    needs re-rounding.
    """
    d = cpc.load()
    for key in ('STD', 'WAL'):
        c = cpc.derive(d, key)
        n = len(c['t'])
        core = np.median([fh.hsc_age_spread(d[f'{key}_temp_T'][i], c['z'],
                                            fh.AGE_ISO_CORE_C)[0]
                          for i in range(n)])
        full = np.median([fh.hsc_age_spread(d[f'{key}_temp_T'][i], c['z'])[0]
                          for i in range(n)])
        assert core < 0.05 and full < 0.10, (
            f'{key}: the trench column is not half-space cooling -- run '
            f'median isotherm disagreement {100*core:.1f} % over 600-1000 C '
            f'and {100*full:.1f} % over 400-1100 C, against the 5 / 10 % '
            f'bounds. Its effective age is then not well defined.')
        a_ridge = np.nanmedian(effective_age(d, key, 'R'))
        assert a_ridge < 3.0, (
            f'{key}: the ridge-column null failed -- the same inversion '
            f'returns {a_ridge:.1f} Myr where the age should be ~0. Either '
            f'the ridge pick or the archive constants have moved.')


def main():
    if not hasattr(np, 'trapz'):
        np.trapz = np.trapezoid
    d = cpc.load()
    if f'STD_sxx_T' not in d.files:
        raise SystemExit('cache lacks sigma_xx — rebuild column_profiles_cache.py')
    # TWO PANELS since 2026-10-11 (Dan): the NEUTRAL PLANE IS DEPRECATED as a
    # mechanical-thickness proxy for this figure and is referenced only in the
    # SI. What replaced it is a fixed STRENGTH criterion -- the depth below
    # which the carried fibre stress stays under 25 MPa -- which is a single
    # physical stress level for both models rather than a per-snapshot
    # fraction.
    #
    # Why the neutral plane goes. Measured over the 37 retained snapshots:
    # h_np is actually the most STABLE proxy in absolute terms (s.d. 1.4 km
    # STD / 2.6 km WAL), but h = 2 h_np doubles that, giving a 6 km (STD) /
    # 10 km (WAL) swing, and -- the real objection -- its L/h ratio DISAGREES
    # BETWEEN THE RUNS, 0.557 against 0.622. The strength criterion gives
    # 0.503 and 0.516, i.e. one number for both models. A proxy whose
    # calibration depends on which run you measure it in is not a proxy.
    #
    # NO BANDS (Dan, 2026-10-11). The spread of three estimates was being
    # drawn as a horizontal span, which read as an uncertainty on h/2. It was
    # not one -- it was the disagreement between three different definitions.
    # With one definition there is nothing to band, and with no bands there is
    # no reason to split the scatter by model, so panel (b) recombines into a
    # single square (it was split in 2026-09-26 precisely because 37 points
    # with spans each overlapped badly).
    # THE TWO AXES SHARE LEFT AND RIGHT EDGES, and the lower one is square
    # IN INCHES. set_box_aspect alone cannot do this: it squares the panel by
    # shrinking its width inside the row, so the scatter sat inset from the
    # time series above it. Here the position is set explicitly after layout
    # -- same x0 and width as the top panel, height chosen so width and
    # height are equal in inches -- which both aligns the panels and makes
    # the 1:1 line a true 45 degrees (Dan, 2026-10-11).
    FW, FH = 7.3, 10.6
    fig = plt.figure(figsize=(FW, FH))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 2.5], hspace=0.20)
    ax_t = fig.add_subplot(gs[0])
    ax_sc = fig.add_subplot(gs[1])
    # THE ESTIMATE, and the reference it is shown against. 900 C is NOT a
    # second estimate here: it is drawn to show that the strength criterion
    # tracks temperature, which is what one would expect and which the
    # measurement confirms -- correlation +1.00 (STD) / +0.99 (WAL) with a
    # constant +3.0 km offset. It is calibrated to the cap at mid-run
    # (T_MECH_C), so the agreement in MEDIAN is by construction; the
    # agreement THROUGH TIME is not.
    # The legend NAMES the two curves; it does not restate the criterion --
    # that is the caption's job (Dan, 2026-10-11).
    # THE LEGEND DISTINGUISHES THE CURVES, it does not re-describe them.
    # These read 'mechanical thickness $h_m$' and '900 $^\circ$C isotherm'
    # beneath a y-axis already labelled 'Mechanical thickness $h_m$ [km]'
    # and a title already reading 'Mechanical thickness'.
    styles = [('yield_cap', '-', '$h_m$'),
              ('thermal', (0, (1, 2)), '900 $^\circ$C')]
    rows = [('model', 'quantity', 'value')]
    band_hits = {}
    for key in ('STD', 'WAL'):
        col = C[key]
        r = thicknesses(d, key)
        for name, ls, lab in styles:
            ax_t.plot(r['t'], r[name] / 1e3, ls=ls, color=col,
                      lw=2.0 if name == 'yield_cap' else 1.3,
                      alpha=1.0 if name == 'yield_cap' else 0.75)
            rows += [(key, f'h_{name}_median_km', f'{np.nanmedian(r[name])/1e3:.1f}'),
                     (key, f'h_{name}_q1_km', f'{np.nanpercentile(r[name], 25)/1e3:.1f}'),
                     (key, f'h_{name}_q3_km', f'{np.nanpercentile(r[name], 75)/1e3:.1f}')]

        # (b) THE h_m/2 SCALING TEST, one point per snapshot, ONE ESTIMATE:
        #   y  L = trench pull / surface pressure deficit
        #   x  h/2 from the 25 MPa strength criterion
        # No x error bar: the former span was the disagreement between three
        # definitions, not an uncertainty on this one (Dan, 2026-10-11). The
        # triangle definition is still excluded from x -- L = triangle/2
        # identically, so using it would make the test circular.
        a_eff = r['triangle'] / 2          # = |Delta GPE*| / Delta P
        hh = r['yield_cap'] / 2e3
        ax_sc.plot(hh, a_eff / 1e3, 'o', color=col, ms=6, alpha=0.85,
                   markeredgecolor='white', markeredgewidth=0.6, label=key,
                   zorder=4)
        # the hit rate inside the +-10 % band, which is what the panel claims
        frac = a_eff / r['yield_cap']               # a_eff / h_m
        inb = (frac >= BAND_LO) & (frac <= BAND_HI)
        band_hits[key] = (int(inb.sum()), int(inb.size), float(np.nanmedian(frac)))
        print(f'   h_m/2 test on the {YIELD_CAP_MPA:.0f} MPa strength estimate: '
              f'a_eff/(h_m/2) = {np.nanmedian(a_eff / 1e3 / hh):.3f}; '
              f'a_eff/h_m = {np.nanmedian(frac):.3f} '
              f'(range {np.nanmin(frac):.3f}-{np.nanmax(frac):.3f}), inside '
              f'{BAND_LO:.2f}-{BAND_HI:.2f}: {inb.sum()}/{inb.size}')
        ratio = r['arm'] / r['np2']
        rows += [(key, 'strength_mean_thickness_km', f'{np.nanmedian(r["h_strength"])/1e3:.1f}'),
                 (key, 'matching_isotherm_C', f'{np.nanmedian(r["T_strength"]):.0f}'),
                 (key, 'h_yield50_median_km', f'{np.nanmedian(r["yield50"])/1e3:.1f}'),
                 (key, 'h_tp90_median_km', f'{np.nanmedian(r["tp90"])/1e3:.1f}'),
                 (key, 'moment_arm_median_km', f'{np.nanmedian(r["arm"])/1e3:.1f}'),
                 (key, 'moment_arm_over_h_median', f'{np.nanmedian(ratio):.3f}'),
                 (key, 'a_eff_median_km', f'{np.nanmedian(r["triangle"]/2)/1e3:.2f}'),
                 (key, 'a_eff_over_h_np2', f'{np.nanmedian(r["triangle"]/2/r["np2"]):.3f}'),
                 (key, 'a_eff_over_h_yield_cap', f'{np.nanmedian(r["triangle"]/2/r["yield_cap"]):.3f}'),
                 (key, 'a_eff_over_h_thermal900', f'{np.nanmedian(r["triangle"]/2/r["thermal"]):.3f}'),
                 (key, 'a_eff_over_h_strengthmean', f'{np.nanmedian(r["triangle"]/2/r["h_strength"]):.3f}'),
                 # NOT "trench pull". This integral runs to 120 km -- the
                 # FULL LOBE, which is what the triangle and arm calibrations
                 # need -- while the reported trench pull is the balance term
                 # at z_c = 80 km. They differ by 1.5-2 % (2.188 vs 2.147 STD,
                 # 1.797 vs 1.772 WAL; up to 0.10 TN/m in single WAL
                 # snapshots), so a table key called trench_pull_median_TNm
                 # put a SECOND number of that name in the committed tables.
                 # Renamed 2026-10-10; the repo already had this exact
                 # collision once, between cpc.partition and
                 # fig_gpe_partition, and Dan's ruling was that everything
                 # the paper reports uses the fixed z_c.
                 (key, 'trench_lobe_total_to_120km_TNm',
                  f'{np.nanmedian(r["trench_lobe"])/1e12:.3f}'),
                 (key, 'trench_lobe_integration_depth_km', '120'),
                 # a_eff's numerator, and the ratio the audit asked for
                 (key, 'trench_pull_at_zc_median_TNm',
                  f'{np.nanmedian(r["trench_pull_zc"])/1e12:.3f}'),
                 (key, 'a_eff_numerator', 'trench pull at z_c (NOT the 120 km lobe)'),
                 (key, 'lobe_over_zc_ratio',
                  f'{np.nanmedian(r["trench_lobe"])/np.nanmedian(r["trench_pull_zc"]):.4f}'),
                 (key, 'trench_pull_trend_per_10Myr',
                  f'{np.polyfit(r["t"], np.asarray(r["trench_pull_zc"])/1e12, 1)[0]*10:.3f}'),
                 # THE NEUTRAL PLANE, retained in the TABLE ONLY (Dan,
                 # 2026-10-11) -- deprecated as a proxy in this figure and
                 # discussed in the SI. These are the numbers that discussion
                 # needs: the swing of 2 h_np across the run, and the fact
                 # that its L/h calibration disagrees between the runs where
                 # the strength criterion's does not.
                 (key, 'h_np_median_km', f'{np.nanmedian(r["np2"])/2e3:.1f}'),
                 (key, 'h_np_range_km',
                  f'{np.nanmin(r["np2"])/2e3:.1f}-{np.nanmax(r["np2"])/2e3:.1f}'),
                 (key, 'h_np2_swing_km',
                  f'{(np.nanmax(r["np2"])-np.nanmin(r["np2"]))/1e3:.1f}'),
                 (key, 'h_np2_sd_km', f'{np.nanstd(r["np2"])/1e3:.1f}'),
                 (key, 'strength_vs_thermal_offset_km',
                  f'{np.nanmedian((r["yield_cap"]-r["thermal"]))/1e3:+.1f}'),
                 (key, 'strength_vs_thermal_correlation',
                  f'{np.corrcoef(r["yield_cap"], r["thermal"])[0,1]:+.3f}'),
                 (key, 'a_eff_over_half_h_strength',
                  f'{np.nanmedian(r["triangle"]/2/(r["yield_cap"]/2)):.3f}'),
                 # THE BAND TEST -- the statistic the panel now claims, in
                 # place of the regression slope (see BAND_LO/BAND_HI)
                 (key, 'a_eff_over_h_m_median', f'{band_hits[key][2]:.3f}'),
                 (key, 'a_eff_over_h_m_min',
                  f'{np.nanmin(r["triangle"]/2/r["yield_cap"]):.3f}'),
                 (key, 'a_eff_over_h_m_max',
                  f'{np.nanmax(r["triangle"]/2/r["yield_cap"]):.3f}'),
                 (key, 'band_lo_hi_fraction_of_h_m',
                  f'{BAND_LO:.2f}-{BAND_HI:.2f}'),
                 (key, 'snapshots_within_band', f'{band_hits[key][0]}'),
                 (key, 'snapshots_total', f'{band_hits[key][1]}'),
                 (key, 'snapshots_within_band_pct',
                  f'{100*band_hits[key][0]/band_hits[key][1]:.0f}'),
                 # EFFECTIVE THERMAL AGE AT THE TRENCH (conventions §6.3).
                 # Folded in here rather than given a figure of its own
                 # because it is the EXPLANATION of panel (a): h_m grows
                 # through the run because the plate arriving at the trench
                 # is getting older. h_m/sqrt(age) below is constant to 3 %
                 # with the SAME constant in both runs, which is what makes
                 # that more than a restatement of the same curve.
                 *age_rows(d, key, r)]
        print(f'=== {key} === (run medians, km)')
        for name, _, lab in styles:
            print(f'   {lab:34s} {np.nanmedian(r[name])/1e3:5.1f}  '
                  f'(IQR {np.nanpercentile(r[name],25)/1e3:.0f}–{np.nanpercentile(r[name],75)/1e3:.0f})')
        Lk = r['triangle'] / 2
        print(f'   a_eff (effective arm)  {np.nanmedian(Lk)/1e3:5.1f} km')
        print(f'   h_m/2 scaling test  a_eff/h: 2h_np {np.nanmedian(Lk/r["np2"]):.3f}, '
              f'yield_cap {np.nanmedian(Lk/r["yield_cap"]):.3f}, '
              f'thermal900 {np.nanmedian(Lk/r["thermal"]):.3f}, '
              f'strength-mean {np.nanmedian(Lk/r["h_strength"]):.3f}')
        print(f'   centroid of the deficit (a different quantity) '
              f'{np.nanmedian(r["arm"])/1e3:.1f} km')

    # TEXT SIZES RAISED (Dan, 2026-10-15). Panel (b) occupies two thirds
    # of the figure and every label in it was set for a panel a third the
    # size. Panel (a) is the constrained one: its y-label nearly spans the
    # axes, so the symbol comes off it -- the legend directly below already
    # carries $h_m$ -- which buys the larger type at the same width.
    ax_t.set_ylabel('Mechanical thickness  [km]', fontsize=13)
    ax_t.set_xlabel('Model time [Myr]', fontsize=13)
    ax_t.tick_params(labelsize=12)
    # TITLES ARE NOUN PHRASES. These read '(a) mechanical thickness at the
    # trench column, and the 900 C isotherm for reference (navy STD, magenta
    # WAL)' and '(b) the effective arm against h_m/2' -- each one narrating
    # its own panel. The column, the window and the colour key belong in the
    # caption; the axes already say what is plotted (Dan, 2026-10-14).
    ax_t.set_title('(a)  Mechanical thickness', fontsize=13, loc='left')
    ax_t.grid(alpha=0.25, color=C_RULE, lw=0.6)
    ax_t.set_ylim(45, 90)
    for _, ls, lab in styles:
        ax_t.plot([], [], color='0.35', lw=1.6, ls=ls, label=lab)
    ax_t.legend(frameon=False, fontsize=12, ncol=2, loc='upper left')

    # (b) ONE SQUARE, both models. Identical reference lines to the retired
    # split version so the comparison is unchanged.
    lim = np.array([float(x) for x in
                    os.environ.get('SCALE_LIM', '25,45').split(',')])
    # ONE BAND, ONE LINE (Dan, 2026-10-14). This drew 0.45, 0.50 and 0.55
    # h_m as three separate legend entries, which is three readings of one
    # statement. The claim is a_eff = 0.5 h_m to within 10 per cent, so the
    # band IS the claim and the centre line locates it. The upper edge still
    # coincides with the companion's fitted 0.55 h_m; that belongs in the
    # caption, not in a third legend entry.
    # The abscissa is h_m/2, so a ratio of f h_m plots as y = 2 f x.
    ax_sc.fill_between(lim, 2 * BAND_LO * lim, 2 * BAND_HI * lim,
                       color='0.86', lw=0, zorder=0,
                       label=r'$0.5\,h_m \pm 10\,\%$')
    ax_sc.plot(lim, lim, 'k-', lw=1.2, zorder=3, label=r'$0.5\,h_m$')
    # THE HIT RATE IS NOT PRINTED ON THE PANEL. It was, as a three-line
    # block in the upper left -- medians and counts for both runs. Those are
    # caption numbers and they are in the table; a panel that states its own
    # result twice over is doing the caption's job.
    n_in = sum(v[0] for v in band_hits.values())
    n_all = sum(v[1] for v in band_hits.values())
    print(f'   band {BAND_LO:.2f}-{BAND_HI:.2f} h_m: {n_in}/{n_all} '
          f'({100 * n_in / n_all:.0f} %) of snapshots, both runs')
    ax_sc.set_xlim(*lim); ax_sc.set_ylim(*lim)
    # WHOLE-KILOMETRE TICKS. The default locator put 27.5 / 32.5 / 37.5 on a
    # 25-45 range, i.e. half-km labels on a quantity quoted to whole km.
    ticks = np.arange(lim[0], lim[1] + 0.1, 5.0)
    ax_sc.set_xticks(ticks); ax_sc.set_yticks(ticks)
    # h_m, NOT bare h. SYMBOLOGY §2 gives h_m for the mechanical thickness
    # and flags bare h as ambiguous (it is also used for the model plate
    # thickness with a discrete base); the manuscript states the relation
    # this panel tests as a_eff ~ h_m/2 (budget.tex, conclusions.tex).
    # NO CRITERION IN THE AXIS LABEL (Dan, 2026-10-11) -- the 25 MPa
    # threshold is stated in the caption, and repeating it here spends
    # label width on something the reader has already been told.
    ax_sc.set_xlabel('Half the mechanical thickness,  $h_m/2$  [km]',
                     fontsize=15)
    # THE EFFECTIVE ARM, a_eff = |Delta GPE*| / Delta P (budget.tex eq.; the
    # manuscript's own symbol). This axis read '$L$', which SYMBOLOGY §2
    # DEPRECATES for this quantity -- L is RESERVED for domain length --
    # and which the manuscript stopped using on 2026-10-08. Fixed 2026-10-11.
    # THE AXIS NAMES THE QUANTITY (Dan, 2026-10-15). A bare
    # $a_\mathrm{eff}$ means nothing to a reader who has not met it;
    # 'effective arm' is the registered name (SYMBOLOGY §2) and there is
    # room for it twice over.
    ax_sc.set_ylabel(r'Effective arm,  $a_{\mathrm{eff}}$  [km]',
                     fontsize=15)
    ax_sc.tick_params(labelsize=13)
    ax_sc.set_title('(b)  Effective arm of trench pull', fontsize=14,
                    loc='left')
    # WHAT IS MEASURED AND WHAT IS PREDICTED, said once on the panel
    # (Dan, 2026-10-14). Both axes are lengths, so without this the reader
    # has no way to tell that one is a measurement and the other a
    # prediction being tested -- and the 0.5 looks arbitrary rather than
    # inherited. Two lines, in the corner the hit-rate block vacated.
    ax_sc.text(0.035, 0.965,
               'measured      '
               r'$a_{\mathrm{eff}} = |\Delta\mathrm{GPE}^{*}| / \Delta P_T$'
               '\npredicted     '
               r'$h_m/2$   (Sandiford 2026)',
               transform=ax_sc.transAxes, va='top', fontsize=13, zorder=5)
    ax_sc.grid(alpha=0.25, color=C_RULE, lw=0.6, zorder=1)
    # LOWER RIGHT: the hit-rate text now owns the upper left, and the
    # scatter's own trend leaves that corner empty
    ax_sc.legend(frameon=False, fontsize=13, loc='lower right')
    # align the two axes and square the lower one, in inches
    fig.canvas.draw()
    pt = ax_t.get_position()
    w_in = pt.width * FW                      # shared axes width [in]
    h_fr = w_in / FH                          # that width as a height fraction
    ps = ax_sc.get_position()
    ax_sc.set_position([pt.x0, ps.y1 - h_fr, pt.width, h_fr])

    # THE ADOPTED VARIANT OWNS THE PLAIN NAME. With the trench column now
    # the default, the suffix was leaving the paper's Figure 9 at
    # fig_mechanical_thickness_trenchcol.png while the plain filename held
    # the retired max-moment version -- breaking the one-script-one-figure
    # naming the whole figure set depends on. Fixed 2026-10-11.
    suffix = '' if FLEX_COL == 'T' else '_maxmomentcol'
    out = os.path.join(ROOT, 'figures',
                       f'fig_mechanical_thickness{suffix}.png')
    fig.savefig(out, bbox_inches='tight', dpi=220)
    print('written:', out)
    tab = write_table('mechanical_thickness', rows[0], rows[1:],
                      script='fig_mechanical_thickness.py', figure='fig_mechanical_thickness.png',
                      models=('STD', 'WAL'),
                      # EVERY FIELD FORMATTED FROM ITS CONSTANT. 'column' was
                      # the literal 'maximum bending moment', which stopped
                      # being true when FLEX_COL moved to the trench on
                      # 2026-10-11: the figure and CSV changed, the sidecar
                      # did not, and the sidecar is what the manuscript audit
                      # reads to know how a number was measured. A convention
                      # written as a literal is a latent lie.
                      meta={'T_mech_C': T_MECH_C,
                            'yield_cap_MPa': YIELD_CAP_MPA,
                            'yield_abs_MPa': YIELD_ABS_MPA,
                            'flex_column': FLEX_COL,
                            'column': _COLNAME,
                            'deficit_column': 'trench (always; the deficit-based '
                                              'definitions are referenced to x_T '
                                              'whichever flexure column is used)',
                            'trench_lobe_depth_km': 120,
                            'zc_km': cpc.ZC_KM,
                            't_min_Myr': cpc.T_MIN_MYR,
                            'midrun_window_Myr': list(cpc.MIDRUN_MYR),
                            'neutral_plane': 'DEPRECATED as a proxy in this '
                                             'figure (2026-10-11); retained in '
                                             'the table for the SI',
                            'band_fraction_of_h_m': [BAND_LO, BAND_HI],
                            'band_hits_both_runs':
                                f'{sum(v[0] for v in band_hits.values())}/'
                                f'{sum(v[1] for v in band_hits.values())}',
                            'band_upper_provenance':
                                '0.55 h_m is the companion study\'s '
                                'elasto-plastic fit; 0.45 is its mirror about '
                                '0.5, NOT a second published bound',
                            'regression_slope':
                                'NOT reported: the slope of a_eff on h_m/2 '
                                'moves 0.90 -> 1.02 as the integration limit '
                                'goes 80 -> 200 km with no plateau, while the '
                                'median ratio is flat to 2 % over that range'})
    print('written:', tab)

# THE CHECKS RUN AT IMPORT, like the quadrature invariants in
# column_profiles_cache -- a figure that cannot be trusted should fail
# loudly when the module loads, not produce a plausible PNG.
_age_self_test()

if __name__ == '__main__':
    main()
