"""Where is the dissipation? Maps on the native triangulation.

⚠ PROOF OF CONCEPT, NOT A PAPER FIGURE. Writes beside itself.
Companion to energy_balance.py, which gives the integrals; this shows the
spatial distribution behind them.

Plotted with matplotlib's tripcolor on the mesh's OWN triangles, so
nothing is resampled -- the 0.47 km cells at the interface and the 63 km
cells in the deep mantle are each drawn at their own size.

Phi is formed AT THE POINTS (repo_context, "Integrating on the native VTK
mesh", trap 2) and shaded gouraud. Duplicated partition cells overdraw
harmlessly here; they only matter for integrals.

COLUMNS
  1  whole domain, log10 Phi -- the honest answer to "where does the
     energy go", which is: almost everywhere, weakly, plus a few bright
     threads.
  2  the subduction zone, same scale, so the reader can see that the
     bright threads are the interface and the slab.
  3  the SHEAR FRACTION Phi_shear / Phi, using the exact additive split
     Phi = (tau_xx^2 + tau_xz^2)/eta. Red = shear-dominated (channels,
     interface), blue = stretching-dominated (large-scale flow).
"""
import os, glob, warnings, logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import Normalize
import pyvista as pv
warnings.filterwarnings('ignore'); logging.disable(logging.ERROR)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
Y_SURF = 2_900_000.0
VMIN, VMAX = -12.0, -5.0        # log10 Phi [W/m^3], shared by both models


def load(key):
    v = pv.read(sorted(glob.glob(os.path.join(
        ROOT, 'data', f'{key}_RefModel', '*.pvtu')))[0])
    eta = np.asarray(v['NormalSP::Viscosity'])[:, 0]
    e = np.asarray(v['NormalSP::SecondInvariant_StrainRate'])
    st = np.asarray(v['NormalSP::Stress'])
    phi = 4.0 * eta * e ** 2
    phi_s = st[:, 1] ** 2 / eta
    tri = mtri.Triangulation(v.points[:, 0] / 1e3,
                             (Y_SURF - v.points[:, 1]) / 1e3,
                             v.cells_dict[5])
    # ⚠ x_T IS READ FROM THE COMMITTED CACHE, not re-picked here. A first
    # version took the global minimum of the free surface, which landed
    # ~290 km from the cached value: the fore-arc basin is deeper than the
    # trench in this model, so an unguarded global argmin finds the wrong
    # low. That is the trap ANALYSIS_CONVENTIONS §3.1 exists for, and it
    # catches throwaway scripts exactly as readily as real ones.
    cache = np.load(os.path.join(ROOT, 'notebooks', 'outputs',
                                 'column_profiles_deep.npz'))
    t = float(np.asarray(v['NormalSP::Time']).flat[0]) / 31557600.0 / 1e6
    i = int(np.argmin(np.abs(cache[f'{key}_t'] - t)))
    xT = float(cache[f'{key}_xT'][i]) / 1e3
    return tri, phi, phi_s, xT


fig, axes = plt.subplots(2, 3, figsize=(15.5, 7.6),
                         gridspec_kw={'width_ratios': [1.35, 1, 1]})
for r, key in enumerate(('STD', 'WAL')):
    tri, phi, phi_s, xT = load(key)
    lp = np.log10(np.maximum(phi, 1e-30))
    frac = np.clip(phi_s / np.maximum(phi, 1e-30), 0, 1)

    ax = axes[r, 0]
    im = ax.tripcolor(tri, lp, shading='gouraud', cmap='magma',
                      vmin=VMIN, vmax=VMAX, rasterized=True)
    ax.set_xlim(0, 8000); ax.set_ylim(2900, 0)
    ax.axhline(660, color='w', lw=0.8, ls=':', alpha=0.7)
    ax.set_title(f'({"ad"[r]})  {key} — whole domain', fontsize=10.5, loc='left')
    ax.set_ylabel('Depth [km]', fontsize=10)
    if r: ax.set_xlabel('x [km]', fontsize=10)

    ax = axes[r, 1]
    ax.tripcolor(tri, lp, shading='gouraud', cmap='magma',
                 vmin=VMIN, vmax=VMAX, rasterized=True)
    ax.set_xlim(xT - 900, xT + 500); ax.set_ylim(800, 0)
    ax.axvline(xT, color='w', lw=0.9, ls='--', alpha=0.8)
    ax.text(xT, 20, ' trench', color='w', fontsize=8.5, va='top')
    ax.set_title(f'({"be"[r]})  {key} — subduction zone', fontsize=10.5, loc='left')
    if r: ax.set_xlabel('x [km]', fontsize=10)

    ax = axes[r, 2]
    im2 = ax.tripcolor(tri, frac, shading='gouraud', cmap='coolwarm',
                       vmin=0, vmax=1, rasterized=True)
    ax.set_xlim(xT - 900, xT + 500); ax.set_ylim(800, 0)
    ax.axvline(xT, color='k', lw=0.9, ls='--', alpha=0.6)
    ax.set_title(f'({"cf"[r]})  {key} — shear fraction', fontsize=10.5, loc='left')
    if r: ax.set_xlabel('x [km]', fontsize=10)

    for ax in axes[r, :]:
        ax.set_facecolor('0.15')
        ax.tick_params(labelsize=8.5)

cb = fig.colorbar(im, ax=axes[:, :2], fraction=0.02, pad=0.015)
cb.set_label(r'$\log_{10}\,\Phi$   [W m$^{-3}$]', fontsize=10)
cb2 = fig.colorbar(im2, ax=axes[:, 2], fraction=0.045, pad=0.03)
cb2.set_label(r'$\Phi_{\rm shear}/\Phi$', fontsize=10)
fig.suptitle('PROOF OF CONCEPT — viscous dissipation on the native mesh '
             '(not a paper figure)', fontsize=11, color='0.35')
out = os.path.join(HERE, 'dissipation_maps.png')
fig.savefig(out, dpi=170, bbox_inches='tight')
print('written:', out)
