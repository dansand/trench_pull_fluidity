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
                           find_first_isostatic_column, find_ridge_x_flow)

DATA = '/Users/DSAND/DATA/numerical_models/OUTPUTS/'
DX, ZC, Y = 1000.0, 75e3, 2_900_000.0
W = 5                      # +-5 km column window (1 km grid)
T_MIN_MYR = 8.0            # mask spin-up
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
            p, txx, tzz, txz, T, fs, vx, vz = mirror_fields_in_x(p, txx, tzz, txz, T, fs, vx, vz)
            szz = np.nan_to_num(tzz - p)
            fs_top = np.nan_to_num(fs[0, :])
            xT, _ = pick_trench_3step(x, z, p, vx, subducting_side='right')
            ti = int(np.argmin(np.abs(x - xT)))
            iI, _ = find_first_isostatic_column(x, fs_top, xT, ti, DX, seaward_sign=+1)
            # refined to the local topographic crest (conventions §3.4,
            # 2026-09-27). The min-max normalisation removed this figure's
            # dependence on the ridge column's VALUE but not on its
            # POSITION: x_R still terminates the profile and sets the span
            # the extrema are taken over.
            xR, iR = find_ridge_x_flow(x, vx[int(np.argmin(np.abs(z - 10e3)))],
                                       xT, seaward_sign=+1, fs_top=fs_top)
            gpe = -np.trapz(szz, z, axis=0)
            ca = lambda f, j: f[..., max(0, j - W):j + W + 1].mean(axis=-1)
            # normalised profiles on the trench->ridge span
            span = (x >= xT) & (x <= xR)
            xh = (x[span] - xT) / (xR - xT)
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
    d = np.load(cache_path)
    XH = d['XH']
    # PAPER_PLAN W25: this figure's caption quotes numbers and had no table.
    rows = [('model', 'quantity', 'value')]
    sm = lambda f: gaussian_filter1d(f, 3)
    # ONE ROW (Dan, 2026-09-25). The figure was 2x2: the second row
    # repeated the same two curves and differed only in which range band
    # was shaded, which is an expensive way to carry one extra envelope.
    # Both bands are now filled in the same panel, distinguished by grey
    # level -- 'option A' of the four encodings trialled. Note the two
    # ranges nearly coincide in STD; the separation is essentially a WAL
    # feature over 0.6-0.9 of the span, so the encoding does its work in
    # one panel of the two.
    # A short RESIDUAL strip under each panel (Dan, 2026-09-25: this is the
    # headline figure and it must not invite scepticism). The obvious
    # objection to any normalised overlay is "you rescaled both to the same
    # interval, so of course they look alike". The strip answers it
    # directly: min-max normalisation fixes only the two extrema of each
    # profile, so every point in between is free to disagree -- and the
    # residual shows by how much it does not. It is information, not the
    # repetition the old second row carried.
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 5.4), sharex=True,
                             gridspec_kw={'height_ratios': [1.0, 0.34]})
    for j, KEY in enumerate(['STD', 'WAL']):
        g, t = d[f'{KEY}_gpe_n'], d[f'{KEY}_topo_n']
        g_av, t_av = g.mean(axis=0), t.mean(axis=0)
        ax = axes[0, j]
        ax.fill_between(XH, sm(t.min(axis=0)), sm(t.max(axis=0)),
                        color='0.55', alpha=0.45, lw=0,
                        label='topography range (8-80 Myr)')
        ax.fill_between(XH, sm(g.min(axis=0)), sm(g.max(axis=0)),
                        color='0.15', alpha=0.28, lw=0,
                        label=r'$\Delta\mathrm{GPE}^*$ range (8-80 Myr)')
        ax.plot(XH, t_av, 'k--', lw=1.5, label='average topography')
        ax.plot(XH, g_av, 'k-', lw=1.8, label=r'average $\Delta\mathrm{GPE}^*$')
        ax.set_title(KEY)
        ax.legend(fontsize=8, loc='lower right')
        if j:
            ax.tick_params(labelleft=False)

        r = g - t
        axr = axes[1, j]
        axr.fill_between(XH, sm(r.min(axis=0)), sm(r.max(axis=0)),
                         color='0.55', alpha=0.40, lw=0, label='full range')
        axr.plot(XH, r.mean(axis=0), 'k-', lw=1.4, label='mean')
        axr.axhline(0, color='k', lw=0.6)
        axr.set_xlabel(r'$(x - x_T)\,/\,(x_R - x_T)$')
        if j:
            axr.tick_params(labelleft=False)
        print(f'   {KEY}: residual (GPE* - topo), normalised units -- '
              f'mean |r| {np.abs(r.mean(axis=0)).mean():.3f}, '
              f'max |mean r| {np.abs(r.mean(axis=0)).max():.3f}, '
              f'worst single snapshot {np.abs(r).max():.3f}')
        # where the steep near-trench rise ends -- the quantity behind the
        # caption's "within the first ~10 per cent of the span"
        x50 = float(XH[np.argmax(g_av >= 0.50)])
        xpl = float(XH[np.argmax(g_av >= 0.95 * g_av[XH >= 0.20][0])])
        tt = d[f'{KEY}_t']
        rows += [(KEY, 'n_snapshots', str(len(tt))),
                 (KEY, 't_min_Myr', f'{tt.min():.1f}'),
                 (KEY, 't_max_Myr', f'{tt.max():.1f}'),
                 (KEY, 'residual_mean_abs', f'{np.abs(r.mean(axis=0)).mean():.4f}'),
                 (KEY, 'residual_max_of_mean', f'{np.abs(r.mean(axis=0)).max():.4f}'),
                 (KEY, 'residual_worst_snapshot', f'{np.abs(r).max():.4f}'),
                 (KEY, 'span_fraction_at_half_rise', f'{x50:.4f}'),
                 (KEY, 'span_fraction_reaching_plateau', f'{xpl:.4f}')]
    axes[0, 0].set_ylabel('normalised value\n(min = 0, max = 1)')
    axes[1, 0].set_ylabel(r'$\Delta$GPE$^*$ $-$ topo')
    for a in axes[1]:
        a.set_ylim(-0.32, 0.32)
    axes[1, 0].legend(fontsize=7, loc='lower right', ncol=2)
    fig.suptitle(r'Normalised $\Delta\mathrm{GPE}^*$ and topography, '
                 '37 snapshots (8-80 Myr)', y=0.98, fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    print('wrote', fig_path)
    print('wrote', write_table('headline_tracking', rows[0], rows[1:],
                               script='fig_headline_tracking.py',
                               figure='fig_headline_tracking.png',
                               models=('STD', 'WAL'),
                               meta={'normalisation': 'per profile, min-max over the '
                                                      'trench-to-ridge span, after a '
                                                      '+-5 km window smooth',
                                     'residual': 'GPE*_n - topo_n, normalised units',
                                     'note': 'closes PAPER_PLAN W25'}))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--recompute', action='store_true')
    a = ap.parse_args()
    if a.recompute or not CACHE.exists():
        compute()
    render(CACHE, FIG)
