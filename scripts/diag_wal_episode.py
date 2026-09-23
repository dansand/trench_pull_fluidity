"""diag_wal_episode — WAL's late loading-and-release episode, against STD as control.

DIAGNOSTIC (diag_, not a manuscript figure). Writes
figures/diag_wal_episode.png and tables/diag_wal_episode.csv.

Dan's reading, 2026-09-24, which this figure tests: for most of both runs
there is a quasi-static balance in which the subducting plate carries
excess GPE and the slab resists it, with the velocity set by the slab and
only a small modulation passing through Delta N_D -- the §1.8 exchange.
WAL follows that for roughly half the run. Then the trench begins to
ADVANCE, and what follows is not ordinary buckling: a large simultaneous
spike in trench depth, plate velocity and slab descent rate, and a return
to trench retreat.

Measured, that is right, and the episode separates cleanly into two
phases with different physics.

  PHASE 1, LOADING (~46-62 Myr; trench advancing, rollback < 0)
    slab dip steepens          61 -> 71 deg, PEAKING MID-PHASE at t = 54
                               (endpoint-to-endpoint it reads -2 deg,
                               which is why the table's "change over
                               loading" understates it -- the dip turns
                               over inside the window and the collapse
                               has begun by t = 62)
    trench deepens            1409 -> 2017 m
    Delta GPE* rises          3.18 -> 4.18 TN/m
    Delta N_D rises           1.76 -> 3.11 TN/m      (absorbs it ALL)
    F_B FALLS                 1.41 -> 1.07 TN/m
    plate velocity FLAT       2.48 -> 2.99 cm/yr
    slab descent FLAT         1.73 -> 1.76 cm/yr

  ⚠ The drive grows by 1.00 TN/m and Delta N_D grows by 1.35. Nothing
  accelerates. The plate is STORING the surplus as compression, not
  spending it -- the quasi-static regime Dan describes, seen loading up.

  PHASE 2, RELEASE (~62-70 Myr)
    slab dip COLLAPSES          71 -> 33 deg
    trench depth PEAKS        2665 m at t = 66   (+7.5 sd on the 20-50 Myr run)
    plate velocity PEAKS      4.83 cm/yr at t = 66  (+4.0 sd)
    slab descent PEAKS        3.05 cm/yr at t = 66  (+6.1 sd)
    trench pull PEAKS         2.82 TN/m at t = 68   (+14.7 sd)
    DEEP descent PEAKS        1.19 cm/yr at t = 68  (+8.3 sd)
    F_B rises                 1.07 -> 1.72 TN/m
    rollback flips           -0.28 -> +1.17 cm/yr, retreat resumes
    Delta N_D stops rising and turns over at t = 64

⚠ THREE THINGS MARK THIS AS NOT ORDINARY BUCKLING.

  1. **The drive is not constant.** §1.8's cycle is an exchange between
     Delta N_D and F_B at a drive that does not move (amplitude
     0.038 TN/m). Here Delta GPE* rises by 56 % and then discharges. The
     buckling cycle redistributes resistance; this episode changes the
     driving term itself.
  2. **The deep sinking rate breaks its cap.** Cerpa et al. (2022) report
     the slab tip holding ~1 cm/yr independent of folding, and we measure
     0.95 cm/yr median with almost no variance -- yet it reaches
     1.19 cm/yr here, +8.3 sd. The lower mantle participates, which in
     the ordinary regime it does not.
  3. **The trench ADVANCES first.** Rollback is negative through the whole
     loading phase. In STD it never is (0 of 37 snapshots).

⚠ THE INTERNAL ORDER OF THE RELEASE IS NOT RESOLVED. Trench depth, plate
velocity and slab descent rate ALL peak at t = 66, within one 2 Myr
sample of each other. So the data support the SEQUENCE (load while
advancing -> release -> retreat resumes) but NOT a causal ordering inside
the release, and in particular do not establish that the deepening trench
drives the slab or the reverse. What does lag unambiguously is the return
to retreat: rollback peaks at t = 74, eight samples after the release.
Any wording must respect that limit.

STD IS THE CONTROL AND SHOWS NOTHING. Over the same window its largest
excursions are +4 to +6 sd in trench pull, ridge push, Delta N_D and
Delta GPE* -- but all of those peak at t = 76-80, i.e. they are the
ENDPOINTS OF SECULAR TRENDS, not an event. Every kinematic quantity in
STD stays within 1.6 sd of its 20-50 Myr median.

WHAT IT MEANS FOR THE PAPER. WAL carries two regimes, and whole-run WAL
statistics average across them (§5.9). The §1.8 exchange result is a
statement about the MODULATION regime and should be quoted from STD,
with WAL's first half as support. This episode is a separate phenomenon
and deserves either its own treatment or an explicit declared mask, as
the ASPECT 10 Myr event received (ANALYSIS_CONVENTIONS §9).

⚠ Cerpa et al. (2022) place this in their narrative: their third folding
episode (t = 35-55 My) occurs "associated with a stationary trench", and
a fourth fold forms after t = 55 My producing "a peak subducting-plate
velocity of 4.8 cm/yr". We measure 4.83 at t = 66. So the episode is
theirs and is real; what is new here is the force-balance anatomy of it.

DRAFT CAPTION. Anatomy of WAL's late loading-and-release episode (right),
with STD over the same interval as a control (left). Shading marks the
loading phase, during which the trench advances, and the release. From
top: trench depth and trench pull; plate velocity and trench rollback;
slab dip with the upper-mantle and sub-660 km descent rates; and the
three balance terms. During loading the driving term and the
normal-stress-difference resultant rise together while nothing
accelerates; at release the trench depth, plate velocity and slab descent
rate peak simultaneously and trench retreat resumes.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.join(_HERE, 'kinematics'))
import column_profiles_cache as cpc
import slab_geometry_cache as sgc
from fig_budget_time import terms
from tables_io import write_table

ROOT = os.path.dirname(_HERE)
RHO_G = 3300.0 * 9.8
C_RULE = '#BFC3D1'
C_W = '#E7298A'
C_TP = '#0072B2'
C_PLATE = '#1B9E77'
C_ROLL = '#E7298A'
C_DIP = '#7B3294'
C_VZ = '#0072B2'
C_VZD = '0.55'
LOAD = (46.0, 62.0)          # trench advancing (rollback < 0), dip steepening
REL = (62.0, 70.0)           # the release
BASE = (20.0, 50.0)          # reference interval for excursion sizes


def main():
    if not hasattr(np, 'trapezoid'):
        np.trapezoid = np.trapz
    d, s = cpc.load(), sgc.load()
    fig, axes = plt.subplots(4, 2, figsize=(13.0, 12.4), sharex='col')
    rows = [('model', 'quantity', 'value')]

    for col, key in enumerate(('STD', 'WAL')):
        c = cpc.derive(d, key); q = terms(d, key)
        z, zc, tc = c['z'], c['zc'], c['t']
        ts = s[f'{key}_t']; n = min(len(tc), len(ts)); t = tc[:n]
        I = lambda a: np.trapezoid(a[:n, zc], z[zc], axis=1) / 1e12
        # stabilised trench pull (x_I column median-filtered through time,
        # §5.10) -- this episode is exactly where an unstable reference
        # would be most tempting to over-read
        TP = I(d[f'{key}_szz_T']) - I(median_filter(d[f'{key}_szz_I'],
                                                    size=(3, 1), mode='nearest'))
        s0 = (d[f'{key}_szz_T'][:n, 0]
              - 0.5 * (d[f'{key}_szz_T'][:n, 1] - d[f'{key}_szz_T'][:n, 0]))
        wT = s0 / RHO_G
        vp = np.abs(d[f'{key}_vx_TR'][:, c['z'] / 1e3 < 20].mean(axis=1))[:n]
        roll = (np.gradient(d[f'{key}_xT'] / 1e3, tc) / 10.0)[:n]
        dip, vz, vzd = (s[f'{key}_dip'][:n], s[f'{key}_vz_upper'][:n],
                        s[f'{key}_vz_deep'][:n])
        ND, FB, GPE = (q['d_nd'][:n] / 1e12, q['f_b'][:n] / 1e12,
                       q['d_gpe'][:n] / 1e12)

        a0, a1, a2, a3 = (axes[r, col] for r in range(4))
        for ax in (a0, a1, a2, a3):
            ax.axvspan(*LOAD, color='#FFE9B0', alpha=0.45, zorder=0)
            ax.axvspan(*REL, color='#F9C2C2', alpha=0.45, zorder=0)
            ax.grid(alpha=0.25, color=C_RULE, lw=0.6)

        a0.plot(t, wT, '-', color=C_W, lw=2.2, label='trench depth $w_T$')
        a0.set_ylabel('$w_T$ [m]', fontsize=9.5, color=C_W)
        a0.tick_params(axis='y', labelcolor=C_W)
        a0b = a0.twinx()
        a0b.plot(t, TP, '-', color=C_TP, lw=2.0)
        a0b.set_ylabel('trench pull [TN/m]', fontsize=9.5, color=C_TP)
        a0b.tick_params(axis='y', labelcolor=C_TP)
        a0.set_title(key, fontsize=11.5)

        a1.plot(t, vp, '-', color=C_PLATE, lw=2.4, label='plate velocity')
        a1.plot(t, roll, '-', color=C_ROLL, lw=1.8, label='trench rollback')
        a1.axhline(0, color='k', lw=1.0)
        a1.set_ylabel('[cm/yr]', fontsize=9.5)
        a1.legend(frameon=False, fontsize=8, loc='upper left', ncol=2)

        h_dip, = a2.plot(t, dip, '-', color=C_DIP, lw=2.4, label='slab dip')
        a2.set_ylabel('slab dip [deg]', fontsize=9.5, color=C_DIP)
        a2.tick_params(axis='y', labelcolor=C_DIP)
        a2b = a2.twinx()
        h_vz, = a2b.plot(t, vz, '-', color=C_VZ, lw=2.0,
                         label='$v_z$ 200–600 km')
        h_vzd, = a2b.plot(t, vzd, '-', color=C_VZD, lw=1.6,
                          label='$v_z$ below 660 km')
        a2b.set_ylabel('slab $v_z$ [cm/yr]', fontsize=9.5, color=C_VZ)
        a2b.tick_params(axis='y', labelcolor=C_VZ)
        a2.legend([h_dip, h_vz, h_vzd],
                  ['slab dip', '$v_z$ 200–600 km', '$v_z$ below 660 km'],
                  frameon=False, fontsize=8, loc='upper right', ncol=3)

        a3.plot(t, GPE, '-', color='b', lw=2.6, label=r'$\Delta$GPE* (drive)')
        a3.plot(t, ND, '-', color='k', lw=2.0, label=r'$\Delta N_D$')
        a3.plot(t, FB, '-', color='red', lw=2.0, label='$F_B$')
        a3.set_ylabel('Force per unit\ndistance [TN/m]', fontsize=9.5)
        a3.set_xlabel('Model time [Myr]', fontsize=11)
        a3.legend(frameon=False, fontsize=8, loc='upper left', ncol=3)

        # ---- numbers -----------------------------------------------------
        ser = {'trench_depth_m': wT, 'trench_pull_TNm': TP,
               'plate_velocity_cmyr': vp, 'rollback_cmyr': roll,
               'slab_dip_deg': dip, 'slab_vz_upper_cmyr': vz,
               'slab_vz_deep_cmyr': vzd, 'delta_nd_TNm': ND,
               'f_b_TNm': FB, 'drive_TNm': GPE}
        b = (t >= BASE[0]) & (t <= BASE[1])
        wnd = (t >= LOAD[0]) & (t <= REL[1] + 10)
        print(f'\n=== {key}')
        for nm, y in ser.items():
            k = int(np.nanargmax(y[wnd])); tk = t[wnd][k]
            exc = (y[wnd][k] - np.nanmedian(y[b])) / np.nanstd(y[b])
            lo = (t >= LOAD[0]) & (t <= LOAD[1])
            dload = float(y[lo][-1] - y[lo][0])
            print(f'   {nm:22s} peak {y[wnd][k]:8.2f} at t={tk:5.1f}  '
                  f'({exc:+5.1f} sd)   change over loading {dload:+8.2f}')
            rows += [(key, f'{nm}_peak', f'{y[wnd][k]:.3f}'),
                     (key, f'{nm}_peak_time_Myr', f'{tk:.1f}'),
                     (key, f'{nm}_excursion_sd', f'{exc:.2f}'),
                     (key, f'{nm}_change_over_loading', f'{dload:.3f}')]
        adv = ((roll < 0) & (t >= 40)).sum()
        print(f'   snapshots with trench ADVANCE after 40 Myr: {adv}')
        rows.append((key, 'advance_snapshots_after_40Myr', str(int(adv))))

    for row in range(4):
        pass   # twin axes carry different units per model; no shared scaling

    fig.suptitle("WAL's late loading-and-release episode (right), with STD as "
                 "control (left)\nshading: loading / trench advance (yellow), "
                 "release (pink)", fontsize=11.5)
    fig.tight_layout()
    out = os.path.join(ROOT, 'figures', 'diag_wal_episode.png')
    fig.savefig(out, bbox_inches='tight', dpi=200)
    print('written:', out)
    tab = write_table('diag_wal_episode', rows[0], rows[1:],
                      script='diag_wal_episode.py',
                      figure='diag_wal_episode.png', models=('STD', 'WAL'),
                      meta={'loading_window_Myr': list(LOAD),
                            'release_window_Myr': list(REL),
                            'baseline_window_Myr': list(BASE),
                            'trench_pull': 'x_I-stabilised (§5.10)',
                            'note': 'diagnostic, not a manuscript figure'})
    print('written:', tab)


if __name__ == '__main__':
    main()
