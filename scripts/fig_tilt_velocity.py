"""fig_tilt_velocity — the back-tilt force against plate speed and convergence.

Writes figures/fig_tilt_velocity.png and tables/tilt_velocity.csv from the
committed column-profile cache.

SI FIGURE, reinstated 2026-10-10 (Dan). The analysis has always carried
this result and the manuscript does not: nothing in the current text
links the back-tilt force to how fast the plate is moving. The claim is
one of the few places the STD/WAL contrast is mechanistic rather than
merely different, so it is worth a figure -- but note that reinstating it
adds a RESULT, and a result should not first appear in a supplement.

WHAT IT REPLACES. The lineage is cell 23 of the working time-evolution
notebook (`back_tilt_velocity_correlation.png`, rendered 2026-09-14),
which was a 2x2: twin time series on top, scatter coloured by time below.
Dan's call: the scatter earns nothing the time series does not already
show, so only the upper row survives, and the convergence rate joins the
plate speed on the same axis.

WHY THE OLD RENDER'S NUMBERS CANNOT BE QUOTED. It predates the 8 Myr
spin-up cut (it begins at 4 Myr), used 20 snapshots rather than 37, and
was computed at z_c = 75 km -- and F_tilt = |Delta P| z_c depends on z_c
directly, so every value moves. It also predates the ridge crest
refinement, which moved Delta P. Everything here is recomputed from the
current cache.

THE TWO VELOCITIES, and why both belong:
  plate speed   |v_x| averaged above 20 km over the trench-to-ridge span
  convergence   plate speed + trench rollback, -d(x_T)/dt
The channel argument is about shear in the sub-plate layer, which the
plate speed sets. The convergence is shown beside it because it is the
quantity that is nearly the same in both runs -- so if the back-tilt
tracked convergence rather than plate speed, the contrast between the
runs would have nothing to explain it.

Both correlations are reported at LEVEL and in INCREMENTS. The increment
correlation is the discriminating one: two series that share a secular
trend correlate at level whatever the mechanism.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import column_profiles_cache as cpc
from fig_mechanical_thickness import T_MECH_C
from tables_io import write_table

ROOT = os.path.dirname(_HERE)
C_MODEL = {'STD': '#002147', 'WAL': '#E5007D'}
C_TILT = '0.45'
C_RULE = '#BFC3D1'


def rates(d, key):
    """Plate speed, trench rollback and convergence, as fig_convergence_partition.

    One implementation of the kinematics would be better than two; this
    mirrors that script rather than inventing a third definition.
    """
    c = cpc.derive(d, key)
    t, zkm = c['t'], c['z'] / 1e3
    vp = np.abs(d[f'{key}_vx_TR'][:, zkm < 20].mean(axis=1))
    xT = d[f'{key}_xT'] / 1e3
    # ROLLBACK IS THE TRENCH MOVING SEAWARD, and seaward is -x in the
    # model's own frame, so the rate is -d(x_T)/dt. This was +d(x_T)/dt,
    # correct while the analysis was mirrored and x_T increased with time;
    # natively x_T DECREASES (STD 3766 -> 2576 km) and the convergence came
    # out as plate speed MINUS rollback -- 0.09 cm/yr against the 3.40 this
    # script's own docstring quotes. Caught 2026-10-10.
    vt = -np.gradient(xT, t) / 10.0                # km/Myr -> cm/yr
    tilt = cpc.partition(d, key)['tilt'] / 1e12         # TN/m, at the fixed z_c
    return t, vp, vp + vt, tilt


def main():
    d = cpc.load()
    # 2 x 2 (Dan, 2026-10-10): levels on top, both series detrended
    # beneath. The detrended row is the one that carries the argument --
    # two series that share a secular trend correlate at level whatever
    # the mechanism, and here the plate velocity and the tilting force
    # both fall through the run.
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.4), sharex='col',
                             gridspec_kw={'height_ratios': [1.0, 0.8]})
    rows = [('model', 'quantity', 'value')]

    for col, key in enumerate(('STD', 'WAL')):
        t, vp, conv, tilt = rates(d, key)
        cm = C_MODEL[key]
        ax = axes[col]
        # TWO CURVES ONLY (Dan, 2026-10-10). The convergence rate was
        # plotted here too and made the panel unreadable -- three curves
        # on two axes, two of them the same colour. Its correlations are
        # still tabulated, which is where that comparison belongs.
        dt_ = lambda y: y - np.polyval(np.polyfit(t, y, 1), t)
        ax = axes[0, col]
        h1, = ax.plot(t, vp, '-', color=cm, lw=2.0, label='plate velocity')
        ax.grid(alpha=0.25, color=C_RULE, lw=0.6)
        ax.set_title(f'({"ab"[col]})  {key}', fontsize=11, loc='left')

        axt = ax.twinx()
        h2, = axt.plot(t, tilt, '-', color=C_TILT, lw=1.7, marker='.', ms=4,
                       label='tilting force $F_{tilt}$')
        if col == 1:
            axt.set_ylabel('$F_{tilt}$  [TN/m]', fontsize=10, color=C_TILT)
        axt.tick_params(axis='y', colors=C_TILT)
        # the legend has to span BOTH axes, so it is built from the two
        # handles rather than left to either one
        if col == 0:
            ax.legend([h1, h2], [h1.get_label(), h2.get_label()],
                      frameon=False, fontsize=9, loc='upper right')

        # ---- detrended and standardised -------------------------------
        # ONE AXIS, not two. This row asks only whether the two series
        # wiggle together, which is a scale-free question, so each
        # residual is divided by its own standard deviation and both are
        # drawn against the same axis. The amplitudes they lose here are
        # carried by the row above. Removing the twin axis is the point:
        # the reader no longer has to map two scales before reading.
        zs_ = lambda y: dt_(y) / dt_(y).std()
        axd = axes[1, col]
        axd.axhline(0, color='k', lw=0.6)
        axd.plot(t, zs_(vp), '-', color=cm, lw=2.0)
        axd.plot(t, zs_(tilt), '-', color=C_TILT, lw=1.7, marker='.', ms=4)
        axd.grid(alpha=0.25, color=C_RULE, lw=0.6)
        axd.set_xlabel('Model time [Myr]', fontsize=10)
        r_det = float(np.corrcoef(dt_(vp), dt_(tilt))[0, 1])
        axd.text(0.03, 0.06, f'detrended   $r$ = {r_det:+.2f}',
                 transform=axd.transAxes, ha='left', va='bottom',
                 fontsize=9, color='0.25',
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.78,
                           pad=1.8))
        rows.append((key, 'r_detrended_plate_speed', f'{r_det:.3f}'))
        # SPLIT AT 50 Myr (Dan, 2026-10-10). WAL's whole-run +0.40 is an
        # average of a first half as tight as STD's and a second half with
        # no relationship at all -- the run enters a different regime,
        # culminating in the shallow upper-mantle release burst that
        # fig_slab_buoyancy shows peaking at twice the run median. Quoting
        # only the whole-run number misrepresents both halves.
        for lab, m in (('early', t < 50), ('late', t >= 50)):
            rows += [(key, f'r_level_plate_speed_{lab}',
                      f'{np.corrcoef(vp[m], tilt[m])[0, 1]:.3f}'),
                     (key, f'r_detrended_plate_speed_{lab}',
                      f'{np.corrcoef(dt_(vp)[m], dt_(tilt)[m])[0, 1]:.3f}'),
                     (key, f'n_snapshots_{lab}', str(int(m.sum())))]

        # level and increment correlations, for both velocities
        stat = {}
        for nm, u in (('plate_speed', vp), ('convergence', conv)):
            rl = float(np.corrcoef(u, tilt)[0, 1])
            ri = float(np.corrcoef(np.diff(u), np.diff(tilt))[0, 1])
            sl = float(np.polyfit(u, tilt, 1)[0])
            stat[nm] = (rl, ri, sl)
            rows += [(key, f'r_level_{nm}', f'{rl:.3f}'),
                     (key, f'r_increment_{nm}', f'{ri:.3f}'),
                     (key, f'slope_TNm_per_cmyr_{nm}', f'{sl:.3f}')]
        # WHAT THE SENTENCE IN §4.1 RESTS ON. The crossing depth z*(t) is
        # not set by boundary-layer thickening alone: the thickening
        # accounts for its secular trend, and the tilt accounts for its
        # departures from that trend. Both are reported at level and
        # detrended, because the two say different things -- at level
        # everything correlates with everything through the shared trend.
        zs = cpc.partition(d, key)['z_star'] / 1e3
        Tm, zkm = d[f'{key}_temp_M'] - 273.15, d['z'] / 1e3
        h900 = np.array([np.interp(T_MECH_C, Ti, zkm) for Ti in Tm])
        det = lambda y: y - np.polyval(np.polyfit(t, y, 1), t)
        for nm, u in (('f_tilt', tilt), ('plate_speed', vp),
                      ('h_900C', h900)):
            rows += [(key, f'r_zstar_{nm}_level',
                      f'{np.corrcoef(zs, u)[0, 1]:.3f}'),
                     (key, f'r_zstar_{nm}_detrended',
                      f'{np.corrcoef(det(zs), det(u))[0, 1]:.3f}')]
        rows += [(key, 'zstar_trend_km_per_10Myr',
                  f'{np.polyfit(t, zs, 1)[0] * 10:.2f}'),
                 (key, 'plate_speed_median_cmyr', f'{np.median(vp):.2f}'),
                 (key, 'convergence_median_cmyr', f'{np.median(conv):.2f}'),
                 (key, 'f_tilt_median_TNm', f'{np.median(tilt):.3f}'),
                 (key, 'n_snapshots', f'{len(t)}')]
        # the discriminating statistic goes in the panel, not the title
        ax.text(0.03, 0.06,
                f'$r$ = {stat["plate_speed"][0]:+.2f}'
                f'   (increments {stat["plate_speed"][1]:+.2f})',
                transform=ax.transAxes, ha='left', va='bottom', fontsize=9,
                color='0.25', bbox=dict(facecolor='white', edgecolor='none', alpha=0.78, pad=1.8))
        print(f'{key}: plate speed r {stat["plate_speed"][0]:+.2f} '
              f'(inc {stat["plate_speed"][1]:+.2f}), slope '
              f'{stat["plate_speed"][2]:.3f} TN/m per cm/yr | convergence r '
              f'{stat["convergence"][0]:+.2f} (inc {stat["convergence"][1]:+.2f})')

    # ONE VELOCITY SCALE for both runs: the contrast between them is the
    # point, and two axes would hide it. Headroom at the bottom for the
    # correlation annotation.
    axes[0, 0].set_ylabel('Plate velocity [cm/yr]', fontsize=10)
    axes[1, 0].set_ylabel('Detrended,\nin units of s.d.', fontsize=9.5)
    for r in (0, 1):
        lo = min(a.get_ylim()[0] for a in axes[r, :])
        hi = max(a.get_ylim()[1] for a in axes[r, :])
        for a in axes[r, :]:
            a.set_ylim(lo, hi)
    fig.suptitle('The tilting force and plate velocity through time',
                 fontsize=10.5, color='0.35')
    fig.tight_layout()
    out = os.path.join(ROOT, 'figures', 'fig_tilt_velocity.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print('written:', out)
    tab = write_table('tilt_velocity', rows[0], rows[1:],
                      script='fig_tilt_velocity.py',
                      figure='fig_tilt_velocity.png', models=('STD', 'WAL'),
                      meta={'zc_km': cpc.ZC_KM,
                            'plate_speed': '|v_x| above 20 km, trench-to-ridge mean',
                            'convergence': 'plate speed + trench rollback -d(x_T)/dt',
                            'F_tilt': '|Delta P| * z_c (conventions §6b.3)',
                            'note': 'increment correlations are the '
                                    'discriminating statistic; level '
                                    'correlations share any secular trend'})
    print('written:', tab)


if __name__ == '__main__':
    main()
