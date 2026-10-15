"""fig_headline_tracking — the paper's headline figure.

Normalised Delta-GPE* and topography vs normalised trench-to-ridge distance,
averaged over every snapshot with t >= 8 Myr, STD and WAL; both min-max range
bands shaded in one panel per model (lightly smoothed). Per-snapshot normalisation: values rescaled trench = 0,
ridge = 1; distance rescaled by the trench-to-ridge span. Columns are +-5 km
means (conventions §4.1); pickers from fluidity_helpers (single implementation).

Usage: python fig_headline_tracking.py [--recompute]
Reads/writes cache ../notebooks/outputs/headline_time_agg.npz; writes
../figures/fig_headline_tracking.png. Greyscale by design (Dan, 2026-09-14).

Draft caption (2026-09-14): Normalised Delta-GPE* and surface topography along
the subducting plate for the Fluidity models STD (left) and WAL (right). At
each snapshot both quantities are rescaled so that the trench column is 0 and
the ridge column is 1, and distance is rescaled by the trench-to-ridge span;
solid and dashed curves are averages over 37 snapshots (t = 8-80 Myr), and
the two shaded bands show the full range through time -- the lighter band the
topography, the darker one Delta-GPE*. The corrected potential-energy resultant tracks
the topography across the entire plate and throughout the run: the driving
topographic pressure gradient is carried jointly by the non-isostatic trench
deflection (the steep rise within the first ~10 percent of the span) and the
isostatic cooling topography that accumulates toward the ridge."""
from pathlib import Path
import sys, glob
import numpy as np
import natsort
import pyvista as pv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tables_io import write_table
from fluidity_helpers import (make_field_extractor, mirror_fields_in_x, pick_trench_3step,
                           find_first_isostatic_column, find_ridge_x_flow,
                           surface_fs, archive_root)

# RESOLVED BUT NOT REQUIRED at import: this script RENDERS from its own
# cache and only needs the archive under --recompute, so demanding the
# directory here would break the render for anyone without the 1.4 GB
# download. Was an ABSOLUTE path under the author's home directory --
# the one file in the repo no other machine could satisfy (2026-10-10).
DATA = archive_root(required=False)
import column_profiles_cache as cpc
DX, ZC, Y = 1000.0, cpc.ZC_KM * 1e3, 2_900_000.0   # conventions §5.1
W = 5                      # +-5 km column window (1 km grid)
from column_profiles_cache import T_MIN_MYR   # the only spin-up cut
XH = np.linspace(0, 1, 201)


_HERE = Path(__file__).resolve().parent
CACHE = _HERE.parent / 'notebooks' / 'outputs' / 'headline_time_agg.npz'
FIG = _HERE.parent / 'figures' / 'fig_headline_tracking.png'

def compute():
    out = {}
    for KEY in ('STD', 'WAL'):
        files = natsort.natsorted(glob.glob(DATA + f'{KEY}_RefModel/subduction_with_LM*.pvtu'))
        rows = []
        for ff in files[1:]:
            v = pv.read(ff)
            v.point_data['p'] = v['NormalSP::Pressure']
            v.point_data['tzz'] = v['NormalSP::Stress'][:, 4]
            v.point_data['txx'] = v['NormalSP::Stress'][:, 0]
            v.point_data['txz'] = -v['NormalSP::Stress'][:, 1]
            v.point_data['T'] = v['NormalSP::Temperature']
            v.point_data['fs'] = v['NormalSP::FreeSurface']
            vel = v['NormalSP::Velocity'] / 3.17098e-10
            v.point_data['vx'] = vel[:, 0]; v.point_data['vy'] = vel[:, 1]
            t_myr = float(np.asarray(v['NormalSP::Time']).flat[0]) / 31557600.0 / 1e6
            if t_myr < T_MIN_MYR:
                continue
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
            # native orientation (W31, 2026-09-28) -- no mirror
            szz = np.nan_to_num(tzz - p)
            # boundary nodes, not the 500 m grid row -- see
            # column_profiles_cache and fluidity_helpers.surface_fs
            fs_top = surface_fs(v, x, mirror_x=False)
            xT, _ = pick_trench_3step(x, z, p, vx, subducting_side='left')
            ti = int(np.argmin(np.abs(x - xT)))
            # x_I IS THE dV/dx ZERO (conventions §3.2, 2026-10-15):
            # the local minimum of the vertical shear resultant, which is
            # blind to compensated topography and so carries none of the
            # cooling-plus-tilt bias the old topographic reference did.
            V_pick = cpc.integrate_to_zc(txz[z <= ZC].T, z[z <= ZC], ZC, axis=1)
            iI = find_first_isostatic_column(x, V_pick, xT,
                                             seaward_sign=-1)
            # refined to the local topographic crest (conventions §3.4,
            # 2026-09-27). The min-max normalisation removed this figure's
            # dependence on the ridge column's VALUE but not on its
            # POSITION: x_R still terminates the profile and sets the span
            # the extrema are taken over.
            xR, iR = find_ridge_x_flow(x, vx[int(np.argmin(np.abs(z - 10e3)))],
                                       xT, seaward_sign=-1, fs_top=fs_top)
            # cell rule, 0 -> z_c (cpc.integrate_to_zc); was trapz over the
            # cell centres, i.e. 0.5 .. ZC-0.5 km
            gpe = -cpc.integrate_to_zc(szz.T, z, ZC, axis=1)
            ca = lambda f, j: f[..., max(0, j - W):j + W + 1].mean(axis=-1)
            # normalised profiles on the trench->ridge span
            span = (x <= xT) & (x >= xR)     # native frame: seaward is -x
            xh = (xT - x[span]) / (xT - xR)  # 0 at the trench, 1 at the ridge
            # MIN-MAX NORMALISATION (Dan, 2026-09-25), replacing the former
            # trench/ridge-column anchoring. Each profile is rescaled to its
            # OWN extrema over the trench-to-ridge span. Three reasons:
            #   1. the old anchors were the trench and ridge COLUMNS, so the
            #      figure inherited every ridge-pick problem -- including the
            #      axial valley, an ~800 m trough only ~15 km wide that the
            #      divergence-based pick lands in by construction;
            #   2. values exceeded 1 wherever a profile overshot its ridge
            #      anchor (6-7 % on the STD mean), which looks like an error;
            #   3. the anchors forced every snapshot to 0 and 1 at fixed x,
            #      so the range band was PINCHED TO ZERO WIDTH at both ends
            #      by construction rather than by the data.
            # Under min-max the extrema sit at different x in different
            # snapshots, so the pinch is smeared rather than imposed.
            # The claim the figure makes is correspondingly narrower and
            # more defensible: the two profiles have the same SHAPE along
            # the plate, not the same amplitude.
            # Profiles are window-smoothed FIRST, with the same +-5 km window
            # the column convention uses (§4.1): taking min/max of a raw
            # profile would anchor on single noisy points, and the old code
            # was inconsistent anyway -- windowed anchors against a raw curve.
            nrm = lambda a: (a - a.min()) / (a.max() - a.min())
            gpe_s = uniform_filter1d(gpe, 2 * W + 1)
            topo_s = uniform_filter1d(gaussian_filter1d(fs_top, 5), 2 * W + 1)
            gpe_n = nrm(gpe_s[span])
            topo_n = nrm(topo_s[span])
            # xh runs 0 (trench) to 1 (ridge), but in the native frame that
            # is DECREASING x, so the arrays come out in descending xh.
            # np.interp requires an increasing abscissa and gives silent
            # nonsense otherwise -- it made the residuals look six times
            # better than they are. Sort before interpolating.
            o = np.argsort(xh)
            xh, gpe_n, topo_n = xh[o], gpe_n[o], topo_n[o]
            # sigma_zz lobe profiles (+-5 km column means)
            d_tp = ca(szz, iI) - ca(szz, ti)       # trench lobe vs depth
            d_rp = ca(szz, iI) - ca(szz, iR)       # ridge lobe vs depth
            w_T = float(ca(topo_s, iI) - ca(topo_s, ti))   # deflection (m, positive = trench below x_I)
            rows.append(dict(t=t_myr, xh=xh, gpe_n=np.interp(XH, xh, gpe_n),
                             topo_n=np.interp(XH, xh, topo_n),
                             d_tp=d_tp, d_rp=d_rp, w_T=w_T, z=z))
            del v, g
        out[KEY] = rows
        print(KEY, len(rows), 'snapshots aggregated (t >=', T_MIN_MYR, 'Myr)')

    np.savez(CACHE,
             **{f'{k}_{q}': np.array([r[q] for r in rows]) for k, rows in out.items()
                for q in ('t', 'gpe_n', 'topo_n', 'd_tp', 'd_rp', 'w_T')},
             z=out['STD'][0]['z'], XH=XH)


def render(cache_path, fig_path):
    """2x2: both distance scales x both models (Dan, 2026-10-12).

    THE TWO ROWS MAKE DIFFERENT ARGUMENTS and neither alone makes both.
    Linear shows that the trench side is 3 % of the plate and still carries
    a comparable share of the signal. Square-root shows what happens inside
    it -- the trench minimum, the rise through the outer rise, the local
    maximum near s/s_R = 0.05 -- which on a linear axis is an unreadable
    spike. It also puts x_I at 0.18 instead of 0.03, i.e. plottable at all.

    THE RESIDUAL ROW IS RETIRED. It answered "do they track?" numerically;
    the square-root row answers it visually AND shows where they stop
    tracking, which is the stronger claim. Its numbers are kept as table
    rows so the tracking claim keeps its quantitative backing.

    Bands by LAYERING, not edges or hatching (both trialled): Delta GPE* is
    the wider range and goes down as the pale fill, the narrower topography
    range nests inside it in darker grey.

    DRAFT CAPTION. Normalised Delta GPE* (solid) and surface topography
    (dashed) along the subducting plate, STD (left) and WAL (right), on a
    linear (a, b) and square-root (c, d) seaward-distance axis. At each
    snapshot both quantities are rescaled so the trench column is 0 and the
    ridge column is 1, and distance by the trench-to-ridge span; curves are
    means over 37 snapshots (t = 8-80 Myr) and the two grey bands the full
    range through time -- the pale band Delta GPE*, the darker one
    topography. The dashed vertical line is the median position of the first
    isostatic column x_I.

    ⚠ THE DOMAIN NAMES DESCRIBE THEIR BOUNDING COLUMNS, NOT THEIR CONTENTS
    (Dan, 2026-10-12). "Ridge side" does NOT mean the topography there is
    everywhere isostatic -- the outer-rise forebulge lies within it, plainly
    visible in (c) and (d). The distinction is that the ridge side is
    bounded, to close approximation, by TWO isostatic columns (x_I and the
    ridge), whereas the trench side is bounded by one isostatic column (x_I)
    and one NON-isostatic column (the trench). That is what makes the
    trench-side difference a net driving force and the ridge-side one a
    difference between two compensated columns.
    """
    d = np.load(cache_path)
    XH = d['XH']
    cc = cpc.load()
    rows = [('model', 'quantity', 'value')]
    # ⚠ NO RENDER-STAGE SMOOTHING (Dan, 2026-10-12). This was
    # gaussian_filter1d(f, 3), and at a boundary where the profile rises
    # steeply from zero the filter pulls in values from the rise and lifts
    # the endpoint: the trench read 0.163 (STD) / 0.152 (WAL) when the
    # cached data is 0.000 and the minimum sits at the trench in 100 % of
    # snapshots. The headline figure was misrepresenting its own
    # normalisation at the single point that defines it. The profiles are
    # already window-smoothed at cache build (+-5 km, conventions §4.1,
    # plus a 5-point Gaussian on the surface), so this was double-smoothing
    # and purely cosmetic. Dan spotted it by asking why the curves did not
    # reach zero at the trench.
    sm = lambda f: f
    C_T, C_R = '#0072B2', '#D55E00'          # domain colours; TEXT only
    TICKS_SQ = [0, 0.02, 0.1, 0.25, 0.5, 1.0]
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.0), sharey=True)

    for j, KEY in enumerate(('STD', 'WAL')):
        g, t = d[f'{KEY}_gpe_n'], d[f'{KEY}_topo_n']
        g_av, t_av = sm(g.mean(axis=0)), sm(t.mean(axis=0))
        c = cpc.derive(cc, KEY)
        m = c['t'] >= cpc.T_MIN_MYR
        xT, xI, xR = cc[f'{KEY}_xT'][m], cc[f'{KEY}_xI'][m], cc[f'{KEY}_xR'][m]
        fI = (xT - xI) / (xT - xR)
        fI_med = float(np.median(fI))
        for i, mode in enumerate(('linear', 'sqrt')):
            ax = axes[i, j]
            X = XH if mode == 'linear' else np.sqrt(XH)
            xi = fI_med if mode == 'linear' else np.sqrt(fI_med)
            ax.fill_between(X, sm(g.min(axis=0)), sm(g.max(axis=0)),
                            facecolor='0.89', lw=0, zorder=1)
            ax.fill_between(X, sm(t.min(axis=0)), sm(t.max(axis=0)),
                            facecolor='0.62', lw=0, alpha=0.75, zorder=2)
            ax.plot(X, t_av, 'k--', lw=1.6, zorder=4, label='topography')
            ax.plot(X, g_av, 'k-', lw=2.1, zorder=4,
                    label=r'$\Delta\mathrm{GPE}^*$')
            # a thin 0.5 rule (Dan, 2026-10-12): the half-way level of the
            # normalisation, which makes it readable at a glance how much
            # of the trench-to-ridge rise is already achieved by x_I
            ax.axhline(0.5, color='0.45', lw=0.7, zorder=3)
            ax.axvline(xi, color='0.2', lw=1.3, ls=(0, (4, 2)), zorder=5)
            ax.text(xi + 0.014, 0.03, '$x_I$', fontsize=16,
                    va='bottom', ha='left')
            # DOMAIN LABELS ON THE SQUARE-ROOT ROW ONLY. On the linear axis
            # the trench side is 3 % of the width; the label needs a leader,
            # which points into empty space and reads as though it names the
            # region under the text. Having room for it is part of what the
            # lower row is for.
            # The domain labels go on ONE panel only -- the lower right
            # (Dan, 2026-10-12). Repeating them on all four is noise; the
            # square-root row is where there is room for them at all.
            # Set on TWO LINES at a larger size (Dan, 2026-10-12): the
            # trench-side column is ~0.18 of the square-root axis, so a
            # one-line label at 9.5 pt had to be small enough to fit the
            # width and read as a minor annotation. Stacked, it fits the
            # narrow side at 12 pt. Both labels are stacked -- they are a
            # matched pair and sizing only one looks like an error.
            if mode == 'sqrt' and j == 1:
                ax.text(xi / 2, 0.985, 'trench\nside', fontsize=14,
                        color=C_T, ha='center', va='bottom', linespacing=1.15,
                        fontweight='bold')
                ax.text((xi + 1) / 2, 0.985, 'ridge\nside', fontsize=14,
                        color=C_R, ha='center', va='bottom', linespacing=1.15,
                        fontweight='bold')
            if mode == 'sqrt':
                ax.set_xticks(np.sqrt(TICKS_SQ))
                ax.set_xticklabels([f'{v:g}' for v in TICKS_SQ], fontsize=12.5)
            ax.set_xlim(0, 1); ax.set_ylim(-0.04, 1.14)
            ax.tick_params(labelsize=12.5)
            ax.grid(alpha=0.22, color=C_RULE if 'C_RULE' in globals()
                    else '#BFC3D1', lw=0.6)
            ax.set_title(f'({"abcd"[i * 2 + j]})  {KEY} — '
                         f'{"linear" if mode == "linear" else "square-root"}'
                         f' scale', fontsize=13, loc='left')
            if i:
                ax.set_xlabel(r'normalised trench–ridge distance  '
                              r'$s\,/\,s_R$', fontsize=14)
            if j:
                ax.tick_params(labelleft=False)

        r = g - t
        x50 = float(XH[np.argmax(g_av >= 0.50)])
        xpl = float(XH[np.argmax(g_av >= 0.95 * g_av[XH >= 0.20][0])])
        tt = d[f'{KEY}_t']
        print(f'   {KEY}: x_I at s/s_R = {fI_med:.4f} (range '
              f'{fI.min():.4f}-{fI.max():.4f}); residual mean |r| '
              f'{np.abs(r.mean(axis=0)).mean():.3f}, worst snapshot '
              f'{np.abs(r).max():.3f}')
        rows += [(KEY, 'n_snapshots', str(len(tt))),
                 (KEY, 't_min_Myr', f'{tt.min():.1f}'),
                 (KEY, 't_max_Myr', f'{tt.max():.1f}'),
                 # the retired residual row's numbers, kept as rows
                 (KEY, 'residual_mean_abs', f'{np.abs(r.mean(axis=0)).mean():.4f}'),
                 (KEY, 'residual_max_of_mean', f'{np.abs(r.mean(axis=0)).max():.4f}'),
                 (KEY, 'residual_worst_snapshot', f'{np.abs(r).max():.4f}'),
                 (KEY, 'span_fraction_at_half_rise', f'{x50:.4f}'),
                 (KEY, 'span_fraction_reaching_plateau', f'{xpl:.4f}'),
                 # x_I, now drawn on the figure
                 (KEY, 'x_I_span_fraction_median', f'{fI_med:.4f}'),
                 (KEY, 'x_I_span_fraction_min', f'{fI.min():.4f}'),
                 (KEY, 'x_I_span_fraction_max', f'{fI.max():.4f}'),
                 (KEY, 's_I_km_min', f'{(xT - xI).min() / 1e3:.0f}'),
                 (KEY, 's_I_km_max', f'{(xT - xI).max() / 1e3:.0f}'),
                 (KEY, 's_R_km_min', f'{(xT - xR).min() / 1e3:.0f}'),
                 (KEY, 's_R_km_max', f'{(xT - xR).max() / 1e3:.0f}')]
    for i in range(2):
        axes[i, 0].set_ylabel('normalised value\n(trench = 0, ridge = 1)',
                              fontsize=14)
    axes[0, 0].legend(frameon=False, fontsize=14, loc='lower right')
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    print('wrote', fig_path)
    print('wrote', write_table(
        'headline_tracking', rows[0], rows[1:],
        script='fig_headline_tracking.py',
        figure='fig_headline_tracking.png', models=('STD', 'WAL'),
        meta={'normalisation': 'per profile, min-max over the trench-to-ridge '
                               'span, after a +-5 km window smooth',
              'x_axis': 'seaward distance s/s_R; rows (a,b) linear, '
                        '(c,d) square-root',
              'x_I': 'median over the retained snapshots; the spread is '
                     'dominated by the RIDGE migrating (s_R 2099-3695 km), '
                     'not by x_I moving (s_I 70-107 km)',
              'residual': 'GPE*_n - topo_n; the residual PANEL was retired '
                          '2026-10-12, these rows replace it',
              't_min_Myr': cpc.T_MIN_MYR}))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--recompute', action='store_true')
    a = ap.parse_args()
    if a.recompute or not CACHE.exists():
        compute()
    render(CACHE, FIG)
