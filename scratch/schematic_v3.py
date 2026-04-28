"""Schematic v3.

Changes from v2:
- horizontal extent compressed (L = 400 km, not 4000)
- vertical exaggeration cranked up: trench depression is the focus
- pre-trench surface uses an elastic-flexure profile (exp·cos), giving the
  characteristic forebulge → down-bend signature of a real subducting plate
- dashed horizontal line at the trench-axis depth, so the depression is
  clearly visible as a "volume of water" between the line and the dipped surface
- PNG only
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# === Geometry parameters (km) ================================================
L         = 400.0    # ridge → trench distance (compressed)
L_slab    = 220.0    # post-bend slab length
S0        = 3.0      # regional cooling subsidence at trench (small, secondary)
W_t       = 14.0     # extra trench depression below regional (BIG, the focus)
alpha     = 70.0     # flexural wavelength (km) — controls forebulge & decay
H_plate   = 50.0     # plate thickness at trench (visible offset between curves)
dip       = np.deg2rad(38)
R_top     = 90.0     # outer (top-surface) bend radius at the trench

# === Pre-trench surface: cooling subsidence + elastic flexure ================
xp = np.linspace(0, L, 600)
z_regional = S0 * np.sqrt(xp / L)
xi = (L - xp) / alpha                                    # distance from trench / α
z_flexure  = W_t * np.exp(-xi) * np.cos(xi)              # → 0 far from trench
z_top_pre  = z_regional + z_flexure

# Plate thickness grows ∝ √age (parallel to surface flexure)
plate_thickness_pre = H_plate * np.sqrt(xp / L)
z_iso_pre = z_top_pre + plate_thickness_pre

# === Bend at the trench (smooth arc → fixed slab dip) ========================
# Tangent at the trench from analytic flexure derivative:
# d/dx [ W_t exp(-xi) cos(xi) ]_{x=L}  =  W_t / α  (xi = (L-x)/α)
dz_dx_at_L  = W_t / alpha + 0.5 * S0 / L
theta_init  = np.arctan(dz_dx_at_L)
bend_angle  = dip - theta_init

# Centre of curvature: perpendicular to initial tangent, on the
# down-into-mantle side. R_top must exceed H_plate so the inner arc
# (R_bot = R_top - H_plate) stays positive.
xC = L - R_top * np.sin(theta_init)
zC = z_top_pre[-1] + R_top * np.cos(theta_init)
R_bot = R_top - H_plate

delta = np.linspace(0, bend_angle, 200)
x_bend_top = xC + R_top * np.sin(theta_init + delta)
z_bend_top = zC - R_top * np.cos(theta_init + delta)
x_bend_iso = xC + R_bot * np.sin(theta_init + delta)
z_bend_iso = zC - R_bot * np.cos(theta_init + delta)

# === Slab continuing at fixed dip from end of bend ===========================
ts = np.linspace(0, L_slab, 300)
x_slab_top = x_bend_top[-1] + ts * np.cos(dip)
z_slab_top = z_bend_top[-1] + ts * np.sin(dip)
x_slab_iso = x_bend_iso[-1] + ts * np.cos(dip)
z_slab_iso = z_bend_iso[-1] + ts * np.sin(dip)

# === Combined ================================================================
x_top = np.concatenate([xp, x_bend_top, x_slab_top])
z_top = np.concatenate([z_top_pre, z_bend_top, z_slab_top])
x_iso = np.concatenate([xp, x_bend_iso, x_slab_iso])
z_iso = np.concatenate([z_iso_pre, z_bend_iso, z_slab_iso])

# Trench axis = surface point at x = L (deepest point of sea-floor exposure)
x_trench_axis = L
z_trench_axis = float(z_top_pre[-1])

# === Plot ====================================================================
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.8,
    "axes.edgecolor": "black",
})

fig, ax = plt.subplots(figsize=(10, 6))

# Trench-axis horizontal reference line
ax.axhline(z_trench_axis, color="k", ls="--", lw=0.7, dashes=(5, 5),
           xmin=0.04, xmax=0.97, zorder=1)
ax.text(L * 0.02, z_trench_axis - 0.6, "trench-axis level",
        fontsize=9, ha="left", va="bottom", style="italic", color="0.25")

# Plate surface (solid) and isotherm (dashed)
ax.plot(x_top, z_top, "k-",  lw=1.7, zorder=3)
ax.plot(x_iso, z_iso, "k-",  lw=1.0, dashes=(6, 4), zorder=3)

# Ridge & trench markers — small arrows from above
for x_mark, label in [(0, "ridge"), (L, "trench")]:
    ax.annotate(label, xy=(x_mark, 0), xytext=(x_mark, -7.5),
                ha="center", va="bottom", fontsize=12,
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color="k",
                                shrinkA=0, shrinkB=2))

# Region labels
lith_x = L * 0.55
lith_z = 0.5 * (z_top_pre[int(0.55 * len(xp))] + z_iso_pre[int(0.55 * len(xp))])
ax.text(lith_x, lith_z, "lithosphere", fontsize=11, style="italic",
        ha="center", va="center")

ax.text(L * 0.55, z_iso_pre[int(0.55 * len(xp))] + 22, "asthenosphere",
        fontsize=11, style="italic", ha="center")

slab_label_x = 0.5 * (x_slab_top[len(ts) // 2] + x_slab_iso[len(ts) // 2])
slab_label_z = 0.5 * (z_slab_top[len(ts) // 2] + z_slab_iso[len(ts) // 2])
ax.text(slab_label_x + 35, slab_label_z + 12, "slab",
        fontsize=11, style="italic", ha="left", va="center")

# Isotherm leader
iso_pt_idx = int(0.55 * len(xp))
ax.annotate("isotherm",
            xy=(xp[iso_pt_idx], z_iso_pre[iso_pt_idx]),
            xytext=(xp[iso_pt_idx] - 130, z_iso_pre[iso_pt_idx] + 36),
            fontsize=10, ha="left",
            arrowprops=dict(arrowstyle="-", lw=0.5))

# === Axes ====================================================================
ax.invert_yaxis()
ax.set_xlim(-30, x_top.max() + 30)
ax.set_ylim(z_iso.max() + 20, -12)
ax.set_xlabel("distance from ridge  [km]")
ax.set_ylabel("depth  [km]   (vertical scale heavily exaggerated)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

out = Path("/tmp/schematic_v3.png")
fig.savefig(out, bbox_inches="tight", dpi=200)
print(f"wrote {out}")
