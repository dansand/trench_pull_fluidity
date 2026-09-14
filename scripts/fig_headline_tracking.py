"""fig_headline_tracking — the paper's headline figure.

Normalised Delta-GPE* and topography vs normalised trench-to-ridge distance,
averaged over every snapshot with t >= 8 Myr, STD and WAL; min-max range bands
(lightly smoothed). Per-snapshot normalisation: values rescaled trench = 0,
ridge = 1; distance rescaled by the trench-to-ridge span. Columns are +-5 km
means (conventions §4.1); pickers from cerpa_helpers (single implementation).

Usage: python fig_headline_tracking.py [--recompute]
Reads/writes cache ../notebooks/outputs/headline_time_agg.npz; writes
../figures/fig_headline_tracking.png. Greyscale by design (Dan, 2026-09-14).

Draft caption (2026-09-14): Normalised Delta-GPE* and surface topography along
the subducting plate for the Fluidity models STD (left) and WAL (right). At
each snapshot both quantities are rescaled so that the trench column is 0 and
the ridge column is 1, and distance is rescaled by the trench-to-ridge span;
solid and dashed curves are averages over 37 snapshots (t = 8-80 Myr), and
grey bands show the full range through time of the topography (top row) and
of Delta-GPE* (bottom row). The corrected potential-energy resultant tracks
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
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cerpa_helpers import (make_field_extractor, mirror_fields_in_x, pick_trench_3step,
                           find_first_isostatic_column, find_ridge_x)

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
            xR, iR = find_ridge_x(x, fs_top, xT, seaward_sign=+1)
            gpe = -np.trapz(szz, z, axis=0)
            ca = lambda f, j: f[..., max(0, j - W):j + W + 1].mean(axis=-1)
            # normalised profiles on the trench->ridge span
            span = (x >= xT) & (x <= xR)
            xh = (x[span] - xT) / (xR - xT)
            gpe_n = (gpe[span] - ca(gpe, ti)) / (ca(gpe, iR) - ca(gpe, ti))
            topo_s = gaussian_filter1d(fs_top, 5)
            topo_n = (topo_s[span] - ca(topo_s, ti)) / (ca(topo_s, iR) - ca(topo_s, ti))
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
    sm = lambda f: gaussian_filter1d(f, 3)
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.5), sharex=True, sharey=True)
    for j, KEY in enumerate(['STD', 'WAL']):
        g, t = d[f'{KEY}_gpe_n'], d[f'{KEY}_topo_n']
        g_av, t_av = g.mean(axis=0), t.mean(axis=0)
        ax = axes[0, j]
        ax.fill_between(XH, sm(t.min(axis=0)), sm(t.max(axis=0)),
                        color='0.82', lw=0, label='topography range (8-80 Myr)')
        ax.plot(XH, t_av, 'k--', lw=1.5, label='average topography')
        ax.plot(XH, g_av, 'k-', lw=1.8, label=r'average $\Delta\mathrm{GPE}^*$')
        ax.set_title(KEY)
        ax.legend(fontsize=8, loc='lower right')
        ax = axes[1, j]
        ax.fill_between(XH, sm(g.min(axis=0)), sm(g.max(axis=0)),
                        color='0.82', lw=0, label=r'$\Delta\mathrm{GPE}^*$ range (8-80 Myr)')
        ax.plot(XH, g_av, 'k-', lw=1.8, label=r'average $\Delta\mathrm{GPE}^*$')
        ax.plot(XH, t_av, 'k--', lw=1.5, label='average topography')
        ax.set_xlabel(r'$(x - x_T)\,/\,(x_R - x_T)$')
        ax.legend(fontsize=8, loc='lower right')
    for ax in axes[:, 0]:
        ax.set_ylabel('normalised value\n(trench = 0, ridge = 1)')
    fig.suptitle(r'Normalised $\Delta\mathrm{GPE}^*$ and topography, averaged over 37 snapshots (8-80 Myr)', y=0.98)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    print('wrote', fig_path)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--recompute', action='store_true')
    a = ap.parse_args()
    if a.recompute or not CACHE.exists():
        compute()
    render(CACHE, FIG)
