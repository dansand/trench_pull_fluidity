"""fig_model_evolution — what the two models do, before any force analysis.

Writes figures/fig_model_evolution.png and tables/model_evolution.csv from
model_evolution_cache.py (build that first, ~2 min).

ORIENTATION FIGURE for the Models section (paper-agent spec, 2026-10-09):
the plate, the slab, the 660 km boundary and the contrasting folding
histories, shown before the reader meets a single force. Two rows, one
per model; four columns, one per epoch.

IT REPLACES the six-panel-per-model notebook figure
(notebooks/further_analysis/fluidity_viscosity_evolution.ipynb, renders
viscosity_evolution_6panel_{STD,WAL}.png), which is not usable here for
three reasons: twelve panels across two separate figures is unwieldy for
a model comparison; its 2 and 4 Myr panels fall inside the declared
spin-up cut that every other figure excludes; and its time sampling does
not resolve WAL's 62-70 Myr episode. The times here are 8, 32, 56, 68 --
the first retained snapshot, two mid-run epochs, and one inside that
episode.

FRAME. Sampled in the model's native coordinates and DISPLAYED MIRRORED,
trench at the left with distance increasing seaward, per
ANALYSIS_CONVENTIONS §1.1. The flip is the single `XMAX - x` below.

ABSOLUTE COORDINATES, NOT TRENCH-REGISTERED, and that is a deliberate
choice. Registering each panel on its own trench would make the columns
more comparable but would hide the trench migration, which is one of the
clearest contrasts between the runs: STD rolls back 1190 km over the run
against WAL's 696. The trench is marked in every panel instead.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, 'kinematics'))
import model_evolution_cache as mec
import column_profiles_cache as cpc
import slab_geometry_cache as sgc
from fluidity_helpers import depth_ticks
from tables_io import write_table

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Crameri batlow_r, as the notebook figure used: perceptually uniform and,
# critically here, it separates the stiff LOWER MANTLE from the stiff SLAB.
# A single-hue ramp (YlGnBu was tried) puts both at the dark end and the
# slab disappears into the 660 background.
try:
    import cmcrameri.cm as cmc
    CMAP = cmc.batlow_r
except ImportError:                      # stand-in, same ordering
    CMAP = plt.get_cmap('RdYlBu_r')
XMAX_KM = 8000.0
LOG_ETA = (18.0, 24.0)
T_MANTLE_K = 1573.15
T_SLAB_K = 0.90 * T_MANTLE_K     # ~1416 K, the sharp slab edge
Z660 = 660.0
# DEPTH LIMIT SET BY THE DATA (Dan, 2026-10-14). This was 1500 km while the
# deepest 1416 K contour in any of the eight panels is 1278 km (WAL, 68 Myr;
# STD reaches 1072). About 15 per cent of every panel was empty mantle.
# The panels are 1:1, so the axes cannot be stretched vertically to open up
# the lithosphere without the dip and fold distortion that the equal aspect
# exists to avoid; what trimming the range buys instead is the SAME 1:1
# scale at a larger magnification for the same figure height.
# Deepest contours, measured: STD 682/848/1002/1072, WAL 702/948/1168/1278 km
# at 8/32/56/68 Myr.
ZLIM = (1320.0, 0.0)
# 1:1 ASPECT (Dan, 2026-10-09). The window is narrowed to 2300 km so that
# equal aspect still leaves the panels readable: at the old 3795 km width
# the figure was 2.08x vertically exaggerated, which distorts slab dip and
# fold geometry -- the two things this figure exists to show.
XLIM = (3300.0, 5600.0)
V_SCALE_KM_PER_CMYR = 150.0   # arrow length: one common scale, all 8 panels
V_ARROW_TAIL_KM = 5420.0      # fixed tail, so lengths compare by eye
V_ARROW_Z_KM = -135.0          # ABOVE the plate, in the margin: black is
                              # invisible against the dark plate band
VZ_BAND_KM = (250.0, 450.0)   # where the slab sinking rate is averaged
VZ_ARROW_X_KM = 3420.0        # mid-upper mantle, clear of the slab
V_REF_CMYR = 2.0              # the reference rate the slab is compared to


def main():
    d = mec.load()
    # display coordinate: mirrored, so seaward increases rightward
    xd = (XMAX_KM - d['x'] / 1e3)[::-1]
    zk = d['z'] / 1e3
    # 4 ROWS x 2 COLUMNS, not 2 x 4 (Dan, 2026-10-09). At 1:1 aspect the
    # 2 x 4 layout is 2.4:1 overall, so on a portrait page it renders at
    # about 1.6 x 1.0 in per panel -- unusable. Transposed, the figure is
    # taller than wide and fills a page column at ~3.3 x 2.2 in per panel.
    # Both comparisons survive the rotation: DOWN a column is one model
    # through time, ACROSS a row is STD against WAL at the same epoch.
    fig, axes = plt.subplots(4, 2, figsize=(7.6, 10.4), sharex=True, sharey=True,
                             layout='constrained')
    rows = [('model', 'quantity', 'value')]
    im = None

    cp = cpc.load()
    for r, k in enumerate(('STD', 'WAL')):
        t = d[f'{k}_t']
        xT = (XMAX_KM - d[f'{k}_xT'] / 1e3)
        # trailing-plate speed at each epoch: the 0-30 km mean of the
        # plate-averaged v_x, the same quantity the kinematics figures use
        tc = cp[f'{k}_t']; vx = cp[f'{k}_vx_TR']
        vpl = np.array([abs(vx[int(np.argmin(np.abs(tc - tt))), :30].mean())
                        for tt in t])
        sg = sgc.load(); zs = sg['z_prof']
        _b = (zs >= VZ_BAND_KM[0]*1e3) & (zs < VZ_BAND_KM[1]*1e3)
        _ts = sg[f'{k}_t']
        _vz = (sg[f'{k}_dEdz'][:, _b].sum(1) / sg[f'{k}_dFdz'][:, _b].sum(1)) / 3.17098e-10
        vzs = np.array([_vz[int(np.argmin(np.abs(_ts - tt)))] for tt in t])
        rows += [(k, f'v_plate_at_{t[i]:.0f}Myr_cmyr', f'{vpl[i]:.2f}')
                 for i in range(4)]
        rows += [(k, f'v_slab_sink_at_{t[i]:.0f}Myr_cmyr', f'{vzs[i]:.2f}')
                 for i in range(4)]
        for c in range(4):
            ax = axes[c, r]
            eta = d[f'{k}_log_eta'][c][::-1, :].T
            tmp = d[f'{k}_temp'][c][::-1, :].T
            im = ax.pcolormesh(xd, zk, eta, cmap=CMAP, shading='auto',
                               vmin=LOG_ETA[0], vmax=LOG_ETA[1], rasterized=True)
            ax.contour(xd, zk, tmp, levels=[T_SLAB_K], colors='white',
                       linewidths=1.6)
            ax.axhline(Z660, color='0.25', lw=1.0, ls='--')
            # the trench, marked in every panel because it migrates
            # TRENCH: a tick crossing the surface. The floating triangle
            # straddled the top spine and read as an axis decoration rather
            # than a position in the model (Dan, 2026-10-09).
            ax.plot([xT[c], xT[c]], [-55, 85], '-', color='#D55E00', lw=2.4,
                    solid_capstyle='butt', clip_on=False, zorder=7)
            ax.set_title(f'({"abcdefgh"[c*2+r]})  {k}   $t$ = {t[c]:.0f} Myr',
                         fontsize=9.5, loc='left')
            # TRAILING-PLATE VELOCITY, one common scale across all eight
            # panels. The plate moves toward the trench, which is leftward
            # in this display, so the arrow points -X. Only the trailing
            # plate is shown: the trench triangles already carry trench
            # migration, and a second velocity definition would turn an
            # orientation figure into an analytical one.
            vp = float(vpl[c])
            ax.annotate('', xy=(V_ARROW_TAIL_KM - vp * V_SCALE_KM_PER_CMYR,
                                V_ARROW_Z_KM),
                        xytext=(V_ARROW_TAIL_KM, V_ARROW_Z_KM),
                        arrowprops=dict(arrowstyle='-|>,head_width=0.19,head_length=0.60', color='k',
                                        lw=1.7, mutation_scale=14,
                                        shrinkA=0, shrinkB=0,
                                        joinstyle='miter'),
                        annotation_clip=False, zorder=9)
            # SLAB SINKING RATE, anchored ON the slab (Dan, 2026-10-09).
            # The dot marks the midpoint of the band the rate is averaged
            # over, 350 km, at the slab's own horizontal position; the
            # arrow hangs off its right-hand side. Previously it floated at
            # a fixed x with a reference tick under it, which meant nothing
            # to a reader and has been removed -- the scale is in the
            # legend.
            zmid = 0.5 * (VZ_BAND_KM[0] + VZ_BAND_KM[1])
            jz = int(np.argmin(np.abs(zk - zmid)))
            cold = np.where(tmp[jz, :] < T_SLAB_K)[0]
            xs = float(np.mean(xd[cold])) if len(cold) else VZ_ARROW_X_KM
            vz = float(vzs[c])
            # WHITE for the sinking-rate annotation (Dan, 2026-10-09): it
            # sits inside the field, where black competes with the slab.
            ax.plot([xs], [zmid], 'o', ms=4.5, color='white', zorder=9)
            # OFFSET WELL CLEAR OF THE SLAB (Dan, 2026-10-09). The arrow
            # used to run down through the slab limb it describes; there is
            # open mantle to the right in every panel.
            ax.annotate('', xy=(xs + 330, zmid + vz * V_SCALE_KM_PER_CMYR),
                        xytext=(xs + 330, zmid),
                        arrowprops=dict(arrowstyle='-|>,head_width=0.19,head_length=0.60', color='white',
                                        lw=1.7, mutation_scale=14,
                                        shrinkA=0, shrinkB=0,
                                        joinstyle='miter'), zorder=9)
            # LEVEL WITH THE DOT, not with the arrow tip: the number
            # labels the sampling point, and a label that slides up and
            # down with the arrow length is harder to compare across panels.
            ax.text(xs + 375, zmid, f'{vz:.1f}', color='white', fontsize=10,
                    ha='left', va='center', zorder=9)
            ax.text(V_ARROW_TAIL_KM - vp * V_SCALE_KM_PER_CMYR - 25,
                    V_ARROW_Z_KM, f'{vp:.1f}', color='k', fontsize=10,
                    ha='right', va='center', clip_on=False, zorder=9)
            # NO FIXED ASPECT. Equal aspect over a 3800 x 1500 km window
            # renders the panels too flat to read the slab in, and the
            # vertical exaggeration that fixes that is better set by the
            # figure box than by a magic number here. The notebook figure
            # made the same trade.
            if c == 3:
                ax.set_xlabel('distance  [km]', fontsize=10)
            if r == 0:
                ax.set_ylabel('Depth [km]', fontsize=10)
        rows += [(k, f'x_trench_at_{t[c]:.0f}Myr_km', f'{xT[c]:.0f}')
                 for c in range(4)]
        rows.append((k, 'trench_migration_over_window_km',
                     f'{xT[-1] - xT[0]:.0f}'))
        print(f'{k}: trench {xT[0]:.0f} -> {xT[-1]:.0f} km '
              f'({xT[-1]-xT[0]:+.0f} km seaward over {t[0]:.0f}-{t[-1]:.0f} Myr)')

    # ---- LOCATOR INSET (Dan, 2026-10-14) -----------------------------
    # The panels show a 2300 x 1320 km window of an 8000 x 2900 km box, and
    # nothing on the figure said so. A reader meeting these panels first has
    # no way to know whether they are looking at the whole model or a corner
    # of it. The inset answers that and nothing else: the full box, the
    # 660 km interface, the slab limb, and the window these panels draw.
    #
    # ⚠ IT SHOWS THE SLAB LIMB ONLY, not the plate. The cache spans
    # x = 2402-6197 km of the 8000 km box -- enough to contain the whole
    # slab, which is why a contour of it is honest, but NOT enough to draw
    # the lithosphere, which would terminate at the cache edge and read as
    # a plate that stops in open mantle. The limb is isolated by taking the
    # 1416 K contour below 150 km only.
    # PLACED AT THE LEGEND'S HEIGHT, on the other side (Dan, 2026-10-14).
    # The bbox carries the domain's own 8000:2900 aspect, so the outline is
    # the box rather than a box-shaped frame; at 8 Myr the field to the
    # right of the slab is empty mantle, which is the room this uses.
    ins = axes[0, 1].inset_axes([0.575, 0.555, 0.40, 0.25])
    ins.set_facecolor('white')
    Tw = d['WAL_temp'][0]
    Tw = Tw if Tw.shape[0] == len(zk) else Tw.T
    deep = zk >= 150.0
    ins.contourf(xd, zk[deep], np.flip(Tw[deep], axis=1),
                 levels=[0.0, T_SLAB_K], colors=['#13375e'])
    ins.axhline(Z660, color='0.45', lw=0.7, ls='--')
    ins.add_patch(Rectangle((XLIM[0], ZLIM[1]), XLIM[1] - XLIM[0],
                            ZLIM[0] - ZLIM[1], fill=False, lw=1.1,
                            ls=(0, (3, 2)), ec='k', zorder=5))
    ins.set_xlim(0, XMAX_KM); ins.set_ylim(2900.0, 0.0)
    ins.set_xticks([]); ins.set_yticks([])
    for sp in ins.spines.values():
        sp.set_linewidth(0.9)
    # the scale INSIDE the box -- below it the text falls out of the panel
    ins.text(0.025, 0.08, f'{XMAX_KM:.0f} $\\times$ 2900 km',
             transform=ins.transAxes, ha='left', va='bottom',
             fontsize=6.8, color='0.35')

    axes[0, 0].set_ylim(*ZLIM)
    axes[0, 0].set_xlim(*XLIM)
    for ax in axes.flat:
        ax.set_aspect('equal')
    # HORIZONTAL, at the bottom. On a portrait page a right-hand colorbar
    # takes a whole column and squeezes the panels to nothing.
    cb = fig.colorbar(im, ax=axes, location='bottom', shrink=0.55,
                      aspect=40, pad=0.02)
    cb.set_label(r'$\log_{10}\eta$  [Pa s]', fontsize=10)
    # one legend for the annotations, not repeated eight times
    h = [plt.Line2D([], [], color='white', lw=1.6,
                    path_effects=None),
         plt.Line2D([], [], color='0.25', lw=1.0, ls='--'),
         plt.Line2D([], [], color='#D55E00', lw=2.4),
         plt.Line2D([], [], color='0.25', lw=2.2)]
    # MINIMAL LEGEND (Dan, 2026-10-09). Four words. Everything else --
    # what the isotherm delineates, that v is the plate velocity above and
    # the slab sinking rate below, the scale, the averaging band -- is
    # CAPTION material. A legend that explains itself is a caption in the
    # wrong place, and it blows out the panel it sits in.
    # IN (a), TWO COLUMNS, spanning the lower mantle (Dan, 2026-10-09).
    # At 8 Myr the slab has not reached 660 km in either run, so the whole
    # lower half of (a) is empty field and the legend can use it rather
    # than being squeezed into a corner.
    axes[0, 0].legend(h, [f'{T_SLAB_K:.0f} K', f'{Z660:.0f} km', 'trench', '$v$'],
                      # A light ground, not a frameless legend: two of the
                      # four handles (660 km, v) are dark grey, which is
                      # what they are in the panels, and they disappear
                      # against the lower mantle without it. Recolouring
                      # them white would make the key disagree with the
                      # thing it keys.
                      frameon=True, framealpha=0.82, edgecolor='none',
                      facecolor='white', fontsize=9, ncol=2,
                      columnspacing=1.6, handlelength=2.2,
                      loc='lower center', bbox_to_anchor=(0.5, 0.06))
    # No overriding/trailing-plate labels: caption material (Dan).

    # The weak layer needs no label: the pink band in the WAL row is
    # unmistakable (Dan, 2026-10-09).

    fig.suptitle('Viscosity structure and slab descent', fontsize=10.5, color='0.35')
    out = os.path.join(ROOT, 'figures', 'fig_model_evolution.png')
    depth_ticks(fig)   # 20 km depth ticks (FIGURE_STYLE)
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print('written:', out)
    tab = write_table('model_evolution', rows[0], rows[1:],
                      script='fig_model_evolution.py',
                      figure='fig_model_evolution.png', models=('STD', 'WAL'),
                      meta={'times_Myr': [float(v) for v in d['STD_t']],
                            'frame': 'sampled native, displayed mirrored '
                                     '(trench left, seaward right)',
                            'slab_contour_K': T_SLAB_K,
                            'role': 'Models-section orientation figure'})
    print('written:', tab)


if __name__ == '__main__':
    main()
