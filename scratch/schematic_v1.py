"""Schematic v1: oceanic plate from ridge → trench → slab.

- sqrt-of-age subsidence at the surface
- sqrt-of-age thickening of an isotherm (meets surface at ridge)
- sharp bend at the trench, straight slab at fixed dip
- black-and-white, no fills
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Geometry (km) -----------------------------------------------------------
L        = 4000.0      # ridge → trench distance
L_slab   = 700.0       # along-slab length
H0       = 100.0       # plate thickness at trench (sqrt growth from 0 at ridge)
S0       = 4.0         # surface subsidence at trench (sqrt growth from 0 at ridge)
dip_deg  = 35.0
dip      = np.deg2rad(dip_deg)

# --- Pre-trench segment ------------------------------------------------------
xp = np.linspace(0, L, 400)
ztop_pre = S0 * np.sqrt(xp / L)
ziso_pre = H0 * np.sqrt(xp / L)

# --- Slab segment ------------------------------------------------------------
s = np.linspace(0, L_slab, 200)
xs = L + s * np.cos(dip)
ztop_slab = S0 + s * np.sin(dip)
ziso_slab = ztop_slab + H0

# --- Combined ----------------------------------------------------------------
x_top = np.concatenate([xp, xs])
z_top = np.concatenate([ztop_pre, ztop_slab])

x_iso = np.concatenate([xp, xs])
z_iso = np.concatenate([ziso_pre, ziso_slab])

# --- Plot --------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.8,
    "axes.edgecolor": "black",
})

fig, ax = plt.subplots(figsize=(11, 4.5))

ax.plot(x_top, z_top, "k-", lw=1.5, label="surface")
ax.plot(x_iso, z_iso, "k-", lw=1.0, dashes=(5, 4), label="isotherm")

# Ridge marker (a "v" + crest line)
ax.plot([0, 0], [-25, 0], "k-", lw=0.8)
ax.text(0, -32, "ridge", ha="center", va="bottom", fontsize=11)

# Trench marker
ax.plot([L, L], [-25, S0], "k-", lw=0.8)
ax.text(L, -32, "trench", ha="center", va="bottom", fontsize=11)

# Slab label
sx = L + 0.55 * L_slab * np.cos(dip)
sz = S0 + 0.55 * L_slab * np.sin(dip) + H0 / 2
ax.text(sx + 30, sz, "slab", fontsize=11, ha="left", va="center", style="italic")

# Lithosphere label
ax.text(L * 0.45, H0 * np.sqrt(0.45) / 2 + 5, "lithosphere",
        fontsize=10, style="italic", ha="center")

# Isotherm label (with leader)
ix = 0.78 * L
iz = H0 * np.sqrt(ix / L)
ax.annotate("isotherm", xy=(ix, iz),
            xytext=(ix - 700, iz + 80),
            fontsize=10, ha="left",
            arrowprops=dict(arrowstyle="-", lw=0.5, color="k"))

# Asthenosphere label (mantle below isotherm in pre-trench)
ax.text(L * 0.55, H0 * np.sqrt(0.55) + 90, "asthenosphere",
        fontsize=10, style="italic", ha="center")

# Axes
ax.invert_yaxis()
ax.set_xlim(-250, x_top.max() + 200)
ax.set_ylim(z_iso.max() + 60, -55)
ax.set_xlabel("distance from ridge  [km]")
ax.set_ylabel("depth  [km]")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

out = Path("/tmp/schematic_v1.png")
fig.savefig(out, bbox_inches="tight", dpi=200)
print(f"wrote {out}")
