"""Schematic v2.

Changes over v1:
- subsidence exaggerated 5× so it's visible (this is a schematic, not a slice)
- smooth circular bend at the trench instead of a sharp corner
- truncate slab at ~250 km depth — keep the figure focused
- relocate labels so they don't collide
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Geometry (km) -----------------------------------------------------------
L          = 4000.0      # ridge → trench distance
S0         = 20.0        # surface subsidence at trench (exaggerated for schematic)
H_plate    = 80.0        # plate thickness (between surface and isotherm) at trench
dip_deg    = 35.0
dip        = np.deg2rad(dip_deg)
R_top      = 220.0       # bend radius of top surface at the trench
L_slab     = 350.0       # post-bend straight slab length (along-slab)

# --- Pre-trench segment ------------------------------------------------------
xp = np.linspace(0, L, 400)
ztop_pre = S0 * np.sqrt(xp / L)
ziso_pre = (S0 + H_plate) * np.sqrt(xp / L)

# --- Bend at the trench ------------------------------------------------------
# Centre of curvature is BELOW the surface (so the plate curves downward).
# Top arc: radius R_top, starts at (L, S0) with horizontal tangent.
# Bottom arc: same centre, radius R_top - H_plate (inner side of the bend).
R_bot = R_top - H_plate
xC = L
zC = S0 + R_top

n_bend = 100
delta = np.linspace(0, dip, n_bend)

x_bend_top = xC + R_top * np.sin(delta)
z_bend_top = zC - R_top * np.cos(delta)

x_bend_iso = xC + R_bot * np.sin(delta)
z_bend_iso = zC - R_bot * np.cos(delta)

# --- Post-bend straight slab -------------------------------------------------
# Continue at fixed dip from the end of the bend
ts = np.linspace(0, L_slab, 200)

x_slab_top = x_bend_top[-1] + ts * np.cos(dip)
z_slab_top = z_bend_top[-1] + ts * np.sin(dip)

x_slab_iso = x_bend_iso[-1] + ts * np.cos(dip)
z_slab_iso = z_bend_iso[-1] + ts * np.sin(dip)

# --- Combined ----------------------------------------------------------------
x_top = np.concatenate([xp, x_bend_top, x_slab_top])
z_top = np.concatenate([ztop_pre, z_bend_top, z_slab_top])

x_iso = np.concatenate([xp, x_bend_iso, x_slab_iso])
z_iso = np.concatenate([ziso_pre, z_bend_iso, z_slab_iso])

# --- Plot --------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.8,
    "axes.edgecolor": "black",
})

fig, ax = plt.subplots(figsize=(11, 5))

ax.plot(x_top, z_top, "k-",  lw=1.6)
ax.plot(x_iso, z_iso, "k--", lw=1.0, dashes=(6, 4))

# --- Ridge & trench markers --------------------------------------------------
# Small triangular markers above the surface
for x_mark, label in [(0, "ridge"), (L, "trench")]:
    ax.annotate(label, xy=(x_mark, 0), xytext=(x_mark, -55),
                ha="center", va="bottom", fontsize=12,
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color="k",
                                shrinkA=0, shrinkB=2))

# --- Region labels -----------------------------------------------------------
# "lithosphere" — between top and isotherm, well into the pre-trench region
lith_x = L * 0.55
lith_z = 0.5 * (S0 * np.sqrt(lith_x/L) + (S0 + H_plate) * np.sqrt(lith_x/L))
ax.text(lith_x, lith_z, "lithosphere", fontsize=11, style="italic",
        ha="center", va="center")

# "asthenosphere" — below the isotherm
ax.text(L * 0.55, (S0 + H_plate) * np.sqrt(0.55) + 60, "asthenosphere",
        fontsize=11, style="italic", ha="center")

# "slab" — alongside the dipping section
slab_mid_top_x = x_slab_top[len(ts) // 2]
slab_mid_top_z = z_slab_top[len(ts) // 2]
slab_mid_iso_x = x_slab_iso[len(ts) // 2]
slab_mid_iso_z = z_slab_iso[len(ts) // 2]
slab_label_x = 0.5 * (slab_mid_top_x + slab_mid_iso_x)
slab_label_z = 0.5 * (slab_mid_top_z + slab_mid_iso_z)
# offset slightly to the right of the slab
ax.text(slab_label_x + 90, slab_label_z + 30, "slab",
        fontsize=11, style="italic", ha="left", va="center")

# "isotherm" — leader line to the dashed curve, in mid pre-trench
iso_label_x = 0.45 * L
iso_pt_x = 0.55 * L
iso_pt_z = (S0 + H_plate) * np.sqrt(iso_pt_x / L)
ax.annotate("isotherm", xy=(iso_pt_x, iso_pt_z),
            xytext=(iso_label_x - 600, iso_pt_z + 110),
            fontsize=10, ha="left",
            arrowprops=dict(arrowstyle="-", lw=0.5))

# --- Axes --------------------------------------------------------------------
ax.invert_yaxis()
ax.set_xlim(-300, x_top.max() + 300)
ax.set_ylim(z_iso.max() + 60, -110)

ax.set_xlabel("distance from ridge  [km]")
ax.set_ylabel("depth  [km]   (subsidence vertically exaggerated)")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

out = Path("/tmp/schematic_v2.png")
fig.savefig(out, bbox_inches="tight", dpi=200)
print(f"wrote {out}")
