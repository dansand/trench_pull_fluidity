"""Schematic v4 — zoom in, water-filled trench, slab truncated.

Changes from v3:
- y-axis zoomed to 0-65 km so the trench depression is the dominant feature
- slab segment truncated (it just exits the figure — don't try to show full descent)
- light grey fill between sea level (z=0) and the plate surface = water/ocean
- dashed horizontal line at the trench-axis depth
- forebulge from the elastic flexure curve is visible
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# === Geometry parameters (km) ================================================
L         = 400.0
L_slab    = 75.0
S0        = 4.0
W_t       = 18.0
alpha     = 65.0
H_plate   = 35.0
dip       = np.deg2rad(40)
R_top     = 80.0

# === Pre-trench surface ======================================================
xp = np.linspace(0, L, 600)
z_regional = S0 * np.sqrt(xp / L)
xi = (L - xp) / alpha
z_flexure  = W_t * np.exp(-xi) * np.cos(xi)
z_top_pre  = z_regional + z_flexure

plate_thickness_pre = H_plate * np.sqrt(xp / L)
z_iso_pre  = z_top_pre + plate_thickness_pre

# === Bend at the trench ======================================================
dz_dx_at_L  = W_t / alpha + 0.5 * S0 / L
theta_init  = np.arctan(dz_dx_at_L)
bend_angle  = dip - theta_init

xC = L - R_top * np.sin(theta_init)
zC = z_top_pre[-1] + R_top * np.cos(theta_init)
R_bot = R_top - H_plate

delta = np.linspace(0, bend_angle, 200)
x_bend_top = xC + R_top * np.sin(theta_init + delta)
z_bend_top = zC - R_top * np.cos(theta_init + delta)
x_bend_iso = xC + R_bot * np.sin(theta_init + delta)
z_bend_iso = zC - R_bot * np.cos(theta_init + delta)

# === Slab (short — exits the plotted region) =================================
ts = np.linspace(0, L_slab, 150)
x_slab_top = x_bend_top[-1] + ts * np.cos(dip)
z_slab_top = z_bend_top[-1] + ts * np.sin(dip)
x_slab_iso = x_bend_iso[-1] + ts * np.cos(dip)
z_slab_iso = z_bend_iso[-1] + ts * np.sin(dip)

# === Combined ================================================================
x_top = np.concatenate([xp, x_bend_top, x_slab_top])
z_top = np.concatenate([z_top_pre, z_bend_top, z_slab_top])
x_iso = np.concatenate([xp, x_bend_iso, x_slab_iso])
z_iso = np.concatenate([z_iso_pre, z_bend_iso, z_slab_iso])

z_trench_axis = float(z_top_pre[-1])

# === Plot ====================================================================
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.8,
    "axes.edgecolor": "black",
})

fig, ax = plt.subplots(figsize=(11, 5))

# Water — fill between sea level (z=0) and the plate surface
ax.fill_between(x_top, 0, z_top, where=(z_top > 0), color="#eaeaea", zorder=1)

# Trench-axis horizontal reference
ax.axhline(z_trench_axis, color="k", ls="--", lw=0.8, dashes=(5, 5), zorder=2)
ax.text(L * 0.02, z_trench_axis - 0.6, "trench-axis depth",
        fontsize=9, ha="left", va="bottom", style="italic", color="0.25")

# Surface and isotherm
ax.plot(x_top, z_top, "k-", lw=1.7, zorder=4)
ax.plot(x_iso, z_iso, "k-", lw=1.0, dashes=(6, 4), zorder=4)

# Sea level ticks
ax.plot([-15, x_top.max() + 15], [0, 0], color="0.4", lw=0.4, zorder=3)
ax.text(L * 0.02, -1.0, "sea level", fontsize=9, ha="left", va="bottom",
        style="italic", color="0.25")

# Ridge & trench markers — small annotation arrows from above
for x_mark, label in [(0, "ridge"), (L, "trench")]:
    ax.annotate(label, xy=(x_mark, 0), xytext=(x_mark, -7),
                ha="center", va="bottom", fontsize=12,
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color="k",
                                shrinkA=0, shrinkB=2))

# Region labels
mid_pre = int(0.55 * len(xp))
ax.text(xp[mid_pre], 0.5 * (z_top_pre[mid_pre] + z_iso_pre[mid_pre]),
        "lithosphere", fontsize=11, style="italic", ha="center", va="center")

ax.text(xp[mid_pre], z_iso_pre[mid_pre] + 14, "asthenosphere",
        fontsize=11, style="italic", ha="center", va="center")

# Slab label — pointed off the lower-right edge with arrow
slab_label_x = x_slab_top[-1]
slab_label_z = 0.5 * (z_slab_top[-1] + z_slab_iso[-1])
ax.annotate("slab", xy=(slab_label_x - 6, slab_label_z),
            xytext=(slab_label_x + 30, slab_label_z + 10),
            fontsize=11, style="italic", ha="left", va="center",
            arrowprops=dict(arrowstyle="-", lw=0.5, color="k"))

# Isotherm leader
iso_pt_idx = int(0.45 * len(xp))
ax.annotate("isotherm",
            xy=(xp[iso_pt_idx], z_iso_pre[iso_pt_idx]),
            xytext=(xp[iso_pt_idx] - 70, z_iso_pre[iso_pt_idx] - 14),
            fontsize=10, ha="left",
            arrowprops=dict(arrowstyle="-", lw=0.5))

# === Axes ====================================================================
ax.invert_yaxis()
ax.set_xlim(-30, x_top.max() + 30)
ax.set_ylim(65, -10)
ax.set_xlabel("distance from ridge  [km]")
ax.set_ylabel("depth  [km]   (vertical scale heavily exaggerated)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

out = Path("/tmp/schematic_v4.png")
fig.savefig(out, bbox_inches="tight", dpi=200)
print(f"wrote {out}")
