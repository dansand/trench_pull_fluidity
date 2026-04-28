[NbConvertApp] Converting notebook cerpa_single_step.ipynb to script
#!/usr/bin/env python
# coding: utf-8

# # Force-balance analysis of the Cerpa et al. Fluidity subduction model — **mirrored along x**
# 
# > **TEST NOTEBOOK.** This is a duplicate of `cerpa_single_step.ipynb` with the model
# > mirrored along the x-axis so that subduction matches the MDOODZ convention used
# > in the trench_pull_force notebooks: subducting plate on the **right**, plate
# > velocity negative (leftward) toward the trench, slab dipping toward the left.
# > Coordinate convention is unchanged: x rightward positive, z downward positive.
# 
# This notebook diagnoses the **vertically integrated horizontal force balance** in 2D Cerpa-style subduction simulations run with the Fluidity finite-element code. The framework is the same as in the companion repository [`trench_pull_force`](../../trench_pull_force/), where it is applied to MDOODZ7.0 outputs.
# 
# **Notebook structure.**
# 
# 1. Setup & imports
# 2. Configuration — model, timestep, grid, **mirror flag**
# 3. Helper functions — field extraction, trench pickers, **mirror operator**
# 4. Build the regular grid, extract fields, **apply x-mirror**
# 5. Trench identification (subducting plate on the right after mirror)
# 6. Vertically integrated traction quantities
# 7. Trench topography — shear-stress vs free-surface
# 7B. First isostatic column & 4-panel column profiles
# 8. Force-balance plots — single timestep
# 9. Diagnostics
# 10. Time evolution — loop over timesteps
# 

# ## 1. Setup & imports
# 
# Numerical libraries, PyVista for VTK/PVTU I/O, and minor configuration to silence VTK chatter.
# 

# In[ ]:


#import pyvista as pv
import numpy as np
import glob
import natsort
import matplotlib.pyplot as plt
import h5py
from scipy.integrate import trapz, cumtrapz
from scipy.ndimage import gaussian_filter
import pyvista as pv
import warnings
import logging
from pathlib import Path

# 1) Redirect VTK C++ error output
pv.set_error_output_file(Path("/tmp/vtk_errors.log"))

# 2) Silence Python logging coming from VTK / PyVista
logging.getLogger("pyvista").setLevel(logging.CRITICAL)
logging.getLogger().setLevel(logging.CRITICAL)

# 3) Silence the Python UserWarning layer
warnings.filterwarnings(
    "ignore",
    message="The VTK reader .*vtkXMLPUnstructuredGridReader.*"
)

plt.rcParams.update({
    "mathtext.fontset": "cm",  
    "font.family": "serif",     
})
import pyvista as pv
#import cmcrameri as cmc


# ## 2. Configuration
# 
# Centralised configuration. Change `MODEL_KEY` to swap between reference models, and `TIMESTEP_INDEX` to load a different snapshot. Grid parameters control the regular grid the PVTU is interpolated onto for vertical integration; both the single-timestep analysis (§4) and the time loop (§9) use these same values.
# 

# In[ ]:


# === Model selection =========================================================
DATA_ROOT = '/Users/DSAND/DATA/numerical_models/OUTPUTS/'

MODELS = {
    'STD': dict(folder='STD_RefModel', modname='subduction_with_LM'),
    'WAL': dict(folder='WAL_RefModel', modname='subduction_with_LM'),
}

MODEL_KEY      = 'WAL'   # 'STD' or 'WAL'
TIMESTEP_INDEX = 5       # 0-indexed; negative values count from the end (-1 = last)

# === Regular-grid parameters (metres) ========================================
DX        = 200.0        # horizontal & vertical grid spacing on the analysis grid
Z_MAX     = 100_000.0    # depth extent
Y_SURFACE = 2_900_000.0  # VTK y-coordinate of the model's free surface

# === Mirror flag =============================================================
# When True, every extracted field is mirrored along x so that subduction
# matches the MDOODZ convention (subducting plate on the right, vx < 0).
# Derived constants below propagate this choice to direction-dependent code.
MIRROR_X        = True
SUBDUCTING_SIDE = "right" if MIRROR_X else "left"
SEAWARD_SIGN    = +1     if MIRROR_X else -1   # use as: tloc + SEAWARD_SIGN * offset_metres

# === Resolved paths ==========================================================
fbase   = DATA_ROOT + MODELS[MODEL_KEY]['folder'] + '/'
modname = MODELS[MODEL_KEY]['modname']

pvtu_files = natsort.natsorted(glob.glob(fbase + modname + '*.pvtu'))
print(f"Model {MODEL_KEY!r}: {len(pvtu_files)} timesteps in {fbase}")
print(f"MIRROR_X = {MIRROR_X}  ->  SUBDUCTING_SIDE = {SUBDUCTING_SIDE!r}, SEAWARD_SIGN = {SEAWARD_SIGN:+d}")


# In[ ]:


#Y_SURFACE


# Pick a single timestep for the analysis below.

# In[ ]:


i = TIMESTEP_INDEX if TIMESTEP_INDEX >= 0 else len(pvtu_files) + TIMESTEP_INDEX
ff = pvtu_files[i]
print(f"Loading timestep {i}: {ff.split('/')[-1]}")
vtk_data = pv.read(ff);


# ## 3. Helper functions
# 
# - `get_field(name)` — reshapes a PyVista point-data array onto the regular grid and masks invalid samples.
# - `find_trench_x(...)` — simple trench picker: coarse pressure-minimum guess plus refined `vx` zero-crossing.
# - `pick_trench_3step(...)` — three-stage picker (pressure → velocity → directional pressure search). More robust for snapshots where the surface pressure has multiple local minima; this is the picker used by the time loop.
# 

# In[ ]:


def get_field(name):
    f = interp[name].reshape((nx_pv, nz_pv), order="F")
    f[~valid] = np.nan
    return f.T        # (z, x)


def mirror_fields_in_x(p, txx, tzz, txz, T, fs, vx, vz):
    """Mirror all 2D fields along x so subduction matches MDOODZ orientation.

    Under the x -> -x reflection (z untouched):
      - scalars and diagonals (p, T, sigma_xx, sigma_zz, fs, vz):  np.flip only
      - off-diagonal stress (sigma_xz) and x-velocity (vx):         np.flip + sign flip
    The x-coordinate array is mirror-symmetric about its midpoint, so it does
    not need to change.
    """
    flip = lambda a: np.flip(a, axis=1)
    return (flip(p), flip(txx), flip(tzz), -flip(txz),
            flip(T), flip(fs), -flip(vx), flip(vz))


def find_trench_x(
    x, z, p, vx,
    z_pick_m=5_000.0,          # use shallow depth for robustness (e.g., 5 km)
    window_km=300.0,           # search window half-width around p-based guess
    smooth_km=20.0,            # smooth vx along x before sign-change search
    require_opposite_sign=True # enforce left<0, right>0 (or vice versa)
):
    """
    Return trench x-location (m) using:
      1) coarse guess from min surface pressure
      2) refined pick from vx sign change near surface inside a window
    """

    # ---- pick depth index to evaluate (nearest to z_pick_m)
    iz = int(np.argmin(np.abs(z - z_pick_m)))
    vx1 = vx[iz, :].copy()
    p1  = p[0, :].copy()  # surface pressure for coarse guess

    # ---- coarse guess from pressure minimum (ignore NaNs)
    j0 = int(np.nanargmin(p1))
    x0 = x[j0]

    # ---- define search window around coarse guess
    W = window_km * 1000.0
    inwin = (x >= x0 - W) & (x <= x0 + W)

    xw = x[inwin]
    vxw = vx1[inwin]

    # ---- smooth vx along x (moving average)
    dx = np.median(np.diff(x))
    k = max(1, int((smooth_km * 1000.0) / dx))
    if k % 2 == 0:
        k += 1
    kernel = np.ones(k) / k
    vxws = np.convolve(vxw, kernel, mode="same")

    # ---- find sign-change locations (zero crossings)
    s = np.sign(vxws)
    s[s == 0] = np.nan
    cross = np.where(np.isfinite(s[:-1]) & np.isfinite(s[1:]) & (s[:-1] * s[1:] < 0))[0]

    if cross.size == 0:
        j = int(np.nanargmin(np.abs(vxws)))
        return xw[j], {"method": "min|vx|", "x0": x0, "iz": iz, "x_window": xw, "vx_window": vxw, "vx_smooth": vxws}

    x_cross = []
    for i in cross:
        x1, x2 = xw[i], xw[i+1]
        v1, v2 = vxws[i], vxws[i+1]
        xc = x1 - v1 * (x2 - x1) / (v2 - v1)
        x_cross.append(xc)
    x_cross = np.array(x_cross)

    if require_opposite_sign:
        keep = []
        for i, xc in zip(cross, x_cross):
            left = vxws[i]
            right = vxws[i+1]
            if np.sign(left) != np.sign(right):
                keep.append(True)
            else:
                keep.append(False)
        keep = np.array(keep, dtype=bool)
        if keep.any():
            x_cross = x_cross[keep]

    jbest = int(np.argmin(np.abs(x_cross - x0)))
    trench_x = float(x_cross[jbest])

    return trench_x, {"method": "vx_zero_cross", "x0": x0, "iz": iz, "x_cross": x_cross,
                      "x_window": xw, "vx_window": vxw, "vx_smooth": vxws}


# In[ ]:


import numpy as np

def moving_average_1d(y, k):
    if k <= 1:
        return y
    if k % 2 == 0:
        k += 1
    kernel = np.ones(k) / k
    return np.convolve(y, kernel, mode="same")

def first_local_minimum(y, min_prominence=0.0):
    """
    Return index of first local minimum in y.
    """
    for i in range(1, len(y) - 1):
        if y[i] < y[i-1] and y[i] < y[i+1]:
            if min_prominence <= 0:
                return i
            if (max(y[i-1], y[i+1]) - y[i]) >= min_prominence:
                return i
    return None



def pick_pressure_minimum_windowed(
    x, p_row, x_v,
    side="left",          # "left" means x < x_v ; "right" means x > x_v
    W_km=300.0,           # window width on that side
    expected_offset_km=50.0,   # optional bias (trench ~ x_v - 50 km)
    choose="closest_to_expected",  # or "deepest"
    allow_edge=False      # if False, reject minima at window edges
):
    W = W_km * 1000.0
    expected_offset = expected_offset_km * 1000.0

    if side == "left":
        mask = (x < x_v) & (x >= x_v - W)
        x_expected = x_v - expected_offset
    else:
        mask = (x > x_v) & (x <= x_v + W)
        x_expected = x_v + expected_offset

    xw = x[mask]
    pw = p_row[mask]

    if xw.size < 5:
        raise RuntimeError("Step-3 window too small (need >=5 points).")

    # Drop NaNs explicitly (NaNs kill local-min tests)
    good = np.isfinite(pw)
    xw, pw = xw[good], pw[good]
    if xw.size < 5:
        raise RuntimeError("Too many NaNs in step-3 window.")

    # Identify strict local minima (no smoothing)
    # local min if p[i] < p[i-1] and p[i] < p[i+1]
    is_min = (pw[1:-1] < pw[:-2]) & (pw[1:-1] < pw[2:])
    idx = np.where(is_min)[0] + 1   # shift because we tested 1:-1

    if idx.size == 0:
        # Fallback: choose minimum of pw, but it may be an edge point
        j = int(np.argmin(pw))
        return float(xw[j]), {
            "method": "fallback_argmin",
            "x_window": xw, "p_window": pw,
            "x_v": x_v
        }

    # Optionally reject minima too close to window edges
    if not allow_edge:
        idx = idx[(idx > 0) & (idx < len(pw)-1)]
        if idx.size == 0:
            j = int(np.argmin(pw))
            return float(xw[j]), {
                "method": "fallback_argmin_after_edge_reject",
                "x_window": xw, "p_window": pw,
                "x_v": x_v
            }

    # Choose which minimum to use
    if choose == "deepest":
        j = idx[np.argmin(pw[idx])]
        method = "localmin_deepest"
    else:
        # closest to expected x (e.g., x_v - 50 km)
        j = idx[np.argmin(np.abs(xw[idx] - x_expected))]
        method = "localmin_closest_to_expected"

    return float(xw[j]), {
        "method": method,
        "x_window": xw, "p_window": pw,
        "x_candidates": xw[idx],
        "p_candidates": pw[idx],
        "x_expected": x_expected,
        "x_v": x_v
    }

import numpy as np

def pick_trench_3step(
    x, z, p, vx,
    p_depth_m=0.0,            # depth for pressure row (0 -> z[0])
    vx_depth_m=5_000.0,       # depth for velocity diagnostics
    Wp1_km=800.0,             # coarse pressure window
    Wv_km=800.0,              # velocity window
    Wp3_km=300.0,             # FINAL pressure window (directional)
    expected_offset_km=50.0,  # expected trench offset from velocity transition
    smooth_km=20.0,           # smoothing ONLY for velocity
    subducting_side="left"    # "left" or "right" of velocity transition
):
    """
    Three-step non-stateful trench picker:
      1) coarse surface pressure minimum
      2) velocity sign-change (plate boundary)
      3) directional, windowed TRUE pressure minimum
    """

    # ---------------------------
    # Select depth rows
    # ---------------------------
    iz_p  = 0 if p_depth_m == 0.0 else int(np.argmin(np.abs(z - p_depth_m)))
    iz_vx = int(np.argmin(np.abs(z - vx_depth_m)))

    p_row  = p[iz_p, :].copy()
    vx_row = vx[iz_vx, :].copy()

    # ---------------------------
    # STEP 1: coarse pressure minimum
    # ---------------------------
    j0 = int(np.nanargmin(p_row))
    x_p0 = float(x[j0])

    # ---------------------------
    # STEP 2: velocity transition
    # ---------------------------
    # ---------------------------
    # STEP 2: velocity transition
    # ---------------------------
    Wv = Wv_km * 1000.0
    in_v = (x >= x_p0 - Wv) & (x <= x_p0 + Wv)
    
    x_vw  = x[in_v]
    vx_vw = vx_row[in_v]
    
    if x_vw.size < 5:
        raise RuntimeError("Velocity window too small")
    
    # smooth velocity ONLY
    dx = np.median(np.diff(x))
    k = max(1, int((smooth_km * 1000.0) / dx))
    if k % 2 == 0:
        k += 1
    kernel = np.ones(k) / k
    vx_s = np.convolve(vx_vw, kernel, mode="same")
    
    s = np.sign(vx_s)
    s[s == 0] = np.nan
    cross = np.where(
        np.isfinite(s[:-1]) &
        np.isfinite(s[1:]) &
        (s[:-1] * s[1:] < 0)
    )[0]
    
    if cross.size == 0:
        jv = int(np.nanargmin(np.abs(vx_s)))
        x_v = float(x_vw[jv])
        vx_method = "min|vx|"
    else:
        xc = []
        for i in cross:
            x1, x2 = x_vw[i], x_vw[i+1]
            v1, v2 = vx_s[i], vx_s[i+1]
            xc.append(x1 - v1 * (x2 - x1) / (v2 - v1))
        xc = np.array(xc)
    
        # 🔑 NEW RULE: pick leftmost velocity transition
        x_v = float(np.min(xc))
        vx_method = "zero_cross_leftmost"

    # ---------------------------
    # STEP 3: directional, windowed TRUE pressure minimum
    # ---------------------------
    Wp3 = Wp3_km * 1000.0
    expected_offset = expected_offset_km * 1000.0

    if subducting_side == "left":
        mask = (x < x_v) & (x >= x_v - Wp3)
        x_expected = x_v - expected_offset
    else:
        mask = (x > x_v) & (x <= x_v + Wp3)
        x_expected = x_v + expected_offset

    xw = x[mask]
    pw = p_row[mask]

    if xw.size < 5:
        raise RuntimeError("Pressure window too small")

    # remove NaNs
    good = np.isfinite(pw)
    xw, pw = xw[good], pw[good]

    if xw.size < 5:
        raise RuntimeError("Too many NaNs in pressure window")

    # TRUE local minima (no smoothing)
    is_min = (pw[1:-1] < pw[:-2]) & (pw[1:-1] < pw[2:])
    idx = np.where(is_min)[0] + 1

    if idx.size == 0:
        # explicit fallback
        j = int(np.argmin(pw))
        trench_x = float(xw[j])
        p_method = "fallback_global_min"
    else:
        # choose minimum closest to expected offset
        j = idx[np.argmin(np.abs(xw[idx] - x_expected))]
        trench_x = float(xw[j])
        p_method = "local_min_windowed"

    # ---------------------------
    # Diagnostics
    # ---------------------------
    info = {
        "x_p0": x_p0,
        "x_v": x_v,
        "trench_x": trench_x,
        "vx_method": vx_method,
        "p_method": p_method,
        "subducting_side": subducting_side,
        "depths_m": {
            "pressure": z[iz_p],
            "velocity": z[iz_vx]
        },
        "windows_km": {
            "Wp1": Wp1_km,
            "Wv": Wv_km,
            "Wp3": Wp3_km
        },
        "debug": {
            "x_vw": x_vw,
            "vx_s": vx_s,
            "xw": xw,
            "pw": pw,
            "x_candidates": xw[idx] if idx.size else None,
            "p_candidates": pw[idx] if idx.size else None
        }
    }

    return trench_x, info


# ## 4. Build grid and extract fields (single timestep)
# 
# Construct a uniform cell-centred grid over `[x_min, x_max] × [0, Z_MAX]` and sample the PVTU onto it. Stress, pressure, temperature, and velocity are extracted as 2D arrays indexed `[z, x]` with depth positive downward.
# 

# In[ ]:


# === Read PVTU and stash named point-data arrays =============================
vtk_data = pv.read(ff)

vtk_data.point_data["p"]   = vtk_data["NormalSP::Pressure"]
vtk_data.point_data["txx"] = vtk_data["NormalSP::Stress"][:, 0]
vtk_data.point_data["tzz"] = vtk_data["NormalSP::Stress"][:, 4]
vtk_data.point_data["txz"] = -vtk_data["NormalSP::Stress"][:, 1]
vtk_data.point_data["T"]   = vtk_data["NormalSP::Temperature"]
vtk_data.point_data["fs"]  = vtk_data["NormalSP::FreeSurface"]  # surface deflection (m)

vel = vtk_data["NormalSP::Velocity"] / 3.17098e-10  # m/s -> m/yr
vtk_data.point_data["vx"] = vel[:, 0]
vtk_data.point_data["vy"] = vel[:, 1]

# === Define a regular cell-centred analysis grid =============================
x_min, x_max, _, _, _, _ = vtk_data.bounds

nx = int((x_max - x_min) / DX)
nz = int(Z_MAX / DX)

x = x_min + (np.arange(nx) + 0.5) * DX     # metres
z = (np.arange(nz) + 0.5) * DX              # depth, metres (positive down)

X, Z = np.meshgrid(x, z, indexing="xy")
Y    = Y_SURFACE - Z                        # depth -> VTK y

grid = pv.StructuredGrid(X.T, Y.T, np.zeros_like(X.T))

# === Interpolate VTK fields onto the regular grid ============================
interp = grid.sample(vtk_data)

nx_pv, nz_pv, _ = interp.dimensions
valid = interp["vtkValidPointMask"].reshape((nx_pv, nz_pv), order="F").astype(bool)

# === Final canonical-form arrays =============================================
# field[z_index, x_index] lives at (x[x_index], z[z_index])
p   = get_field("p")
txx = get_field("txx")
tzz = get_field("tzz")
txz = get_field("txz")
T   = get_field("T")
vx  = get_field("vx")
vz  = -get_field("vy")            # positive downward
fs  = get_field("fs")             # free-surface deflection (top row only)

# === Apply x-mirror ==========================================================
if MIRROR_X:
    p, txx, tzz, txz, T, fs, vx, vz = mirror_fields_in_x(p, txx, tzz, txz, T, fs, vx, vz)
    print("(applied x-mirror: subduction now right-to-left, vx < 0 in trailing plate)")

print("x (m):", x[0], "->", x[-1], "  nx =", len(x))
print("z (m):", z[0], "->", z[-1], "  nz =", len(z))
print("p.shape =", p.shape)


# In[ ]:


#fs.shape, p.shape


# In[ ]:





# ## 5. Trench identification
# 
# Two pickers are exercised here. The simple one (`find_trench_x`) gives a coarse anchor; the 3-step picker (`pick_trench_3step`) refines it via a directional pressure search on the subducting-plate side.
# 
# *This remains the most fragile step in the workflow — see the visual check at the end of the section if a pick looks wrong.*
# 

# In[ ]:


surface_p = p[10, :]          # surface row
ploc = x[np.argmin(surface_p)]


# In[ ]:


tloc, info = find_trench_x(x, z, p, vx, z_pick_m=0, window_km=800, smooth_km=1)
print("trench_x (km):", tloc/1000, "method:", info["method"])


# In[ ]:


tindx = np.argmin(np.abs(x - tloc))
tindx


# In[ ]:


tloc, info = pick_trench_3step(
    x, z, p, vx,
    vx_depth_m=5000,
    Wp1_km=1000,
    Wv_km=800,
    Wp3_km=800,
    smooth_km=40,
    subducting_side=SUBDUCTING_SIDE,
)

print("Trench (km):", tloc / 1000)
print("Velocity anchor (km):", info["x_v"] / 1000)
print("Methods:", info["vx_method"], info["p_method"])


# In[ ]:


fig, ax = plt.subplots()

ax.plot(x*1e-3 , p[0,:], marker=".")

ax.vlines(tloc*1e-3, 1e8, -1e8, color='r')
ax.vlines(info["x_p0"]*1e-3, 1e8, -1e8, color='g')
ax.vlines(info["x_v"]*1e-3, 1e8, -1e8, color='b')

ax.set_xlim(tloc*1e-3 - 500, tloc*1e-3 + 500)


# ## 6. Vertically integrated traction quantities
# 
# Compute the depth-integrated quantities that enter the static horizontal force balance:
# 
# $$\Delta F_D - \Delta \mathrm{GPE}^* + F_B = 0$$
# 
# - $F_D = \overline{\tau_{xx} - \tau_{zz}}$ — in-plane differential stress resultant (tension-like / compression-like).
# - $\mathrm{GPE}^* = \overline{p - \tau_{zz}} = -\overline{\sigma_{zz}}$ — pressure-component, carries topographic information including the trench-pull contribution. The constant offset is chosen so $\mathrm{GPE}^*(x_T) = F_D(x_T)$, which makes the curves directly comparable in §8.
# - $V = -\overline{\tau_{xz}}$ and $\partial V / \partial x$ — vertical-shear-stress integral. The topography comparison in §7 uses this.
# - $F_B$ — cumulative basal traction (referenced to zero at the trench).
# 

# In[ ]:


from scipy.integrate import cumulative_trapezoid


# In[ ]:


# These are *resultants* — depth-integrated stress components, not directional                                                                                   
# tractions. A resultant only acquires a signed meaning once paired with the
# outward normal of a chosen plane: V(x) is equal and opposite on the two                                                                                        
# faces of a vertical plane (n̂ = +x̂ vs n̂ = −x̂), by Newton's third law, and
# the same caveat applies to Fd, GPE, and FB. The "right − left for face                                                                                         
# deltas, basal integral oriented left → right" convention used downstream
# in §8 is what closes the signed balance ΔF_D − ΔGPE* + F_B = 0.                                                                                                
                                                                                                                                                                 
Fd   = np.trapz(txx - tzz, z, axis=0)                                                                                                                            
GPE_ = -np.trapz((-p + tzz), z, axis=0)                                                                                                                          
GPE  = GPE_ - GPE_[tindx] + Fd[tindx]                                                                                                                            
V    = np.trapz(txz, z, axis=0)
dVdx = np.gradient(V, x)                                                                                                                                         
                                                                                                                                                                 
# σ_xz at the deepest grid row. Becomes a horizontal basal traction (positive
# in +x̂ when σ_xz > 0) once paired with n̂ = +ẑ — the convention used downstream.                                                                                 
tau_b = txz[-1, :]                                                                                                                                               

# Cumulative basal force per unit out-of-plane length (N/m), referenced to                                                                                       
# zero at the trench. Left-to-right integration matches the right − left
# convention used for the face resultants.                                                                                                                       
FB_ = cumulative_trapezoid(tau_b, x, initial=0.0)         
FB  = FB_ - FB_[tindx] 


# In[ ]:


#FB[tindx]


# ## 7. Trench topography — shear-stress support
# 
# Trench topography is **non-isostatic**: the deflection isn't supported by density variations within the column, but by horizontal gradients of the vertical shear stress $\tau_{xz}$. Two reconstructions of topography on the seaward side of the trench should agree if that's true:
# 
# - **Shear-stress-supported topography** (from the integrated shear-stress gradient):
# 
#   $$h_\tau(x) \;=\; -\frac{1}{\rho_m g}\,\frac{\partial V}{\partial x}, \qquad V(x) \;=\; -\!\!\int_0^{z_\mathrm{max}}\!\tau_{xz}(x, z)\,dz$$
# 
# - **Actual topography** from the model's `NormalSP::FreeSurface` field at the top row (the surface deflection itself).
# 
# Both are referenced to a column $\sim 500$ km seaward of the trench (regional, isostatic). For a non-isostatic feature like the trench the two should overlap.
# 
# This anchors the rest of the analysis — the **first isostatic column** seaward of the trench is where the two curves return to zero, i.e. where the shear-stress-supported component has decayed away.
# 

# In[ ]:


from scipy.ndimage import gaussian_filter1d

# Smooth V before differentiating to clean up grid-scale noise.
# σ in grid points; with DX = 1 km, σ = 10 means 10 km smoothing.
V_smooth = gaussian_filter1d(V, sigma=10, mode="nearest")
dVdx_s   = np.gradient(V_smooth, x)

rho_m, g = 3300.0, 9.8

# Reference column: ~500 km seaward of the trench (regional, isostatic)
x_ref_offset_km = 500
#i_ref = int(np.argmin(np.abs(x - (tloc - x_ref_offset_km * 1000))))
i_ref = int(np.argmin(np.abs(x - (tloc + SEAWARD_SIGN * x_ref_offset_km * 1000)))) 

# Shear-stress-supported topography (elevation convention: positive = up)
h_tau = -dVdx_s / (rho_m * g)
h_tau = h_tau - h_tau[i_ref]

# Actual topography from the free-surface field, top row of the grid
h_actual = fs[0, :] - fs[0, i_ref]
#h_actual = gaussian_filter1d(h_actual, sigma=3, mode="nearest")

# Plot range: from the reference column to slightly past the trench
xs_rel_km = (x - tloc) / 1000.0
#mask = (xs_rel_km >= -(x_ref_offset_km + 50)) & (xs_rel_km <= 50)
mask = (xs_rel_km >= -50) & (xs_rel_km <= x_ref_offset_km + 50) 

fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(xs_rel_km[mask], h_actual[mask], "k-", lw=1.7,
        label=r"$h_\mathrm{actual}$  (free-surface field)")
ax.plot(xs_rel_km[mask], 1*h_tau[mask], color="#E5007D", lw=1.4, ls="--",
        label=r"$h_\tau = -(1/\rho_m g)\,\partial V/\partial x$")

ax.axhline(0, color="0.6", lw=0.5)
ax.axvline(0, color="0.6", lw=0.5)
ax.text(2, ax.get_ylim()[1] * 0.92, "trench",
        fontsize=9, ha="left", va="top", color="0.4")

ax.set_xlabel("distance from trench  [km]")
ax.set_ylabel("topography  [m]   (elevation, positive up)")
ax.legend(frameon=False, loc="lower left", fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(alpha=0.25)

fig.tight_layout()

figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)
fig.savefig(figures_dir / "trench_topo_shear_vs_pressure.png",
            bbox_inches="tight", dpi=220)
plt.show()


# In[ ]:





# In[ ]:


fig, ax = plt.subplots()

ax.plot(fs[0,:])


# ## 7B. First isostatic column & 4-panel column profiles
# 
# Seaward of the trench, $V(x) = -\!\int\!\tau_{xz}\,dz$ peaks (driven by the trench depression's flexural support) and decays back to the regional value across the trailing plate. The **first isostatic column** is the first stationary point of $V$ — equivalently the first zero-crossing of $\partial V/\partial x$ — seaward of the trench. We use it as the trailing-plate anchor for the column-difference figure that follows.
# 
# In Cerpa, the subducting plate is on the left, so *seaward* = decreasing $x$. The cell below searches leftward of the trench, within `WINDOW_KM`, for the first sign change in $\partial V/\partial x$. If none is found it falls back to the most-flexed column (max $|V|$).

# In[ ]:


tindx = int(np.argmin(np.abs(x - tloc)))

# Find first isostatic column: first column going seaward from the trench where
# the free surface has recovered to its regional (undeflected) level.
# After x-mirror, "seaward" = increasing x = higher index (trailing plate is on the right).

BUFFER_KM    = 30
WINDOW_KM    = 500
SMOOTH_KM    = 5      # light smoothing on the surface

# Regional-reference column, 500 km seaward of the trench
i_far = int(np.argmin(np.abs(x - (tloc + SEAWARD_SIGN * 500e3))))
h_rel = fs[0, :] - fs[0, i_far]

sigma_grid = max(1, int(SMOOTH_KM * 1000.0 / DX))
h_smooth   = gaussian_filter1d(h_rel, sigma=sigma_grid, mode="nearest")

buffer_grid = int(BUFFER_KM * 1000.0 / DX)
window_grid = int(WINDOW_KM * 1000.0 / DX)

# Walk seaward from the trench (with buffer) and stop where the surface has
# returned to or above regional (h_rel >= 0).  SEAWARD_SIGN sets the direction.
i_start = tindx + SEAWARD_SIGN * buffer_grid
i_end   = tindx + SEAWARD_SIGN * window_grid
i_start = max(0, min(len(x)-1, i_start))
i_end   = max(0, min(len(x)-1, i_end))
step    = SEAWARD_SIGN

xi_indx = None
for i in range(i_start, i_end, step):
    if h_smooth[i] >= 0.0:
        xi_indx = i
        break

if xi_indx is None:
    xi_indx     = i_end
    pick_method = "fallback: end of search window (no topo recovery found)"
else:
    pick_method = f"first topo recovery seaward of trench  (buffer={BUFFER_KM} km, smooth={SMOOTH_KM} km)"

xi_loc  = float(x[xi_indx])
xm_loc  = 0.5 * (tloc + xi_loc)
xm_indx = int(np.argmin(np.abs(x - xm_loc)))

w_T = float(fs[0, xi_indx] - fs[0, tindx])

print(f"x_T = {tloc/1e3:8.1f} km  (tindx  = {tindx})")
print(f"x_I = {xi_loc/1e3:8.1f} km  (xi_indx = {xi_indx},  delta = {(xi_loc - tloc)/1e3:+7.1f} km)")
print(f"x_m = {xm_loc/1e3:8.1f} km  (xm_indx = {xm_indx},  delta = {(xm_loc - tloc)/1e3:+7.1f} km)")
print()
print(f"fs[0, tindx]   = {fs[0, tindx]:+9.1f} m")
print(f"fs[0, xi_indx] = {fs[0, xi_indx]:+9.1f} m")
print(f"w_T = fs[0, xi_indx] - fs[0, tindx] = {w_T:+.1f} m")
print(f"method: {pick_method}")


# In[ ]:


2477.4 + 0.5*(2574.9 - 2477.4)


# In[ ]:


# Column-placement check — no masks; full arrays plotted, domain set with xlim.
# Top:  shear-stress resultant V(x)              [TN/m]                                                                                                                
# Bot:  pseudo-topography  h_τ = -(1/ρ_m g) dV/dx  [m]    
                                                                                                                                                                       
xkm_rel  = (x - tloc) * 1e-3                              
h_pseudo = -dVdx / (rho_m * g)         # metres (sign matches real topography)                                                                                         
                                                                                                                                                                       
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True,                                                                                                   
                                gridspec_kw=dict(hspace=0.10))                                                                                                         
                                                                                                                                                                       
ax1.plot(xkm_rel, V_smooth * 1e-12, "k-", lw=1.5)                                                                                                                      
ax1.axhline(0, color="0.6", lw=0.5)
ax1.set_ylabel(r"$V(x)$  [TN/m]")                                                                                                                                      
ax1.set_title(f"{MODEL_KEY} snap {TIMESTEP_INDEX}:  "                                                                                                                  
              f"shear-stress resultant and pseudo-topography near the trench")                                                                                         
                                                                                                                                                                       
ax2.plot(xkm_rel, h_pseudo, "k-", lw=1.5)                                                                                                                              
ax2.axhline(0, color="0.6", lw=0.5)                       
ax2.set_xlabel(r"$x - x_\mathrm{trench}$  [km]")                                                                                                                       
ax2.set_ylabel(r"$h_\tau = -(1/\rho_m g)\,\partial V/\partial x$  [m]")                                                                                                


# Three column lines on both panels                                                                                                                                    
columns = [                                               
    (tloc,       r"$x_T$",  "#002147"),                                                                                                                                
    (xi_loc,     r"$x_I$",  "#E5007D"),
    (x[xm_indx], r"$x_m$",  "0.15"),                                                                                                                                   
]                                                                                                                                                                      
for x_col, label, color in columns:
    xrel = (x_col - tloc) * 1e-3                                                                                                                                       
    for ax_ in (ax1, ax2):                                
        ax_.axvline(xrel, color=color, lw=1.4, alpha=0.9)                                                                                                              
    ax1.annotate(label, xy=(xrel, 0), xycoords=("data", "axes fraction"),
                 xytext=(0, 4), textcoords="offset points",                                                                                                            
                 ha="center", va="bottom", color=color, fontsize=11, fontweight="bold")
                                                                                                                                                                       
# 300-km-wide window centred on midpoint of x_T and x_I, applied via xlim
WIN_HW_KM = 150                                                                                                                                                        
xc_rel    = (0.5 * (tloc + xi_loc) - tloc) * 1e-3         
ax1.set_xlim(xc_rel - WIN_HW_KM, xc_rel + WIN_HW_KM)                                                                                                                   
ax1.set_ylim(-2, 2)                                                                                                                                                    
ax2.set_ylim(-2e3, 2e3)                                                                                                                                                                     
ax1.grid(alpha=0.2)                                                                                                                                                    
ax2.grid(alpha=0.2)                                       
fig.tight_layout()
plt.show() 


# In[ ]:


fig, ax = plt.subplots(figsize=(9, 4.2))

ax.plot(xs_rel_km[mask], h_actual[mask], "k-", lw=1.7,
        label=r"$h_\mathrm{actual}$  (free-surface field)")

ax.plot(xs_rel_km[mask], 1*h_tau[mask], color="#E5007D", lw=1.4, ls="--",
        label=r"$h_\tau = -(1/\rho_m g)\,\partial V/\partial x$")

ax.axhline(0, color="0.6", lw=0.5)
ax.axvline(0, color="0.6", lw=0.5)
ax.text(2, ax.get_ylim()[1] * 0.92, "trench",
        fontsize=9, ha="left", va="top", color="0.4")

ax.set_xlabel("distance from trench  [km]")
ax.set_ylabel("topography  [m]   (elevation, positive up)")
ax.legend(frameon=False, loc="lower left", fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(alpha=0.25)

# Three column lines on both panels
columns = [
    (tloc,         r'$x_T$',  '#002147'),
    (xi_loc,       r'$x_I$',  '#E5007D'),
    (x[xm_indx],   r'$x_m$',  '0.15'),
]
for x_col, label, color in columns:
    xrel = (x_col - tloc) * 1e-3
    ax.axvline(xrel, color=color, lw=1.4, alpha=0.9)
    ax.annotate(label, xy=(xrel, 0), xycoords=('data', 'axes fraction'),
                 xytext=(0, 4), textcoords='offset points',
                 ha='center', va='bottom', color=color, fontsize=11, fontweight='bold')

fig.tight_layout()



# ### Four-panel column profiles
# 
# Mirroring `profiles.png` from `trench_pull_force/model_analysis/figures/`. Three columns are sampled: the trench column $x_T$, the first isostatic column $x_I$, and their midpoint $x_m$. Both $x_T$ and $x_I$ are smoothed over $\pm 10\;DX$ to suppress single-column noise.
# 
# - **(a)** $\tau_{xx}-\tau_{zz}$ at $x_m$, with twin top-axis showing $T(z)$
# - **(b)** $\tau_{xz}$ at $x_m$
# - **(c)** column difference $\Delta\sigma_{zz}(z) = \sigma_{zz}(x_I, z) - \sigma_{zz}(x_T, z)$. Shaded area between the curve and zero, integrated to compensation, is the trench-pull contribution. Dashed line marks the analytical scaling $-(\rho_m - \rho_w)\,g\,w_T$ (with $\rho_w = 0$ for the no-water Cerpa setup).
# - **(d)** cumulative $\Delta\mathrm{GPE}^*(z) = -\!\int_0^z(\sigma_{zz}^{(I)} - \sigma_{zz}^{(T)})\,dz'$ in TN/m. Should plateau at the regional offset by compensation depth.
# 
# The Cerpa pressure is dynamic (lithostatic subtracted), but for a column **difference** the lithostatic part cancels, so $\Delta\sigma_{zz}$ from the model output is directly comparable to the analytical scaling.

# In[ ]:


#tloc


# In[ ]:


fs[0, tindx]


# In[ ]:


fs[0, xi_indx] - fs[0, tindx]


# In[ ]:


#z_np_km, w_T, delrho, rho_m 


# In[ ]:


# 4-panel column-profile figure (mirrors trench_pull_force/profiles.png).
# No masks; domain set with xlim. w_T is the simple free-surface difference.

SMOOTHWIN = 1                          # +/- grid columns to average around picked columns
RHO_W     = 0.0                         # Cerpa: no water above the surface

# sigma_zz = -p + tau_zz  (Cerpa pressure is dynamic; lithostatic part cancels in column diff)
sig_zz = -p + tzz

def col_avg(arr, ix, win=SMOOTHWIN):
    lo, hi = max(0, ix - win), min(arr.shape[1] - 1, ix + win)
    return arr[:, lo:hi+1].mean(axis=1)

sig_zz_T     = col_avg(sig_zz, tindx)
sig_zz_I     = col_avg(sig_zz, xi_indx)
delta_sig_zz = sig_zz_I - sig_zz_T
delta_GPE_z  = -cumulative_trapezoid(delta_sig_zz, z, initial=0.0)

# w_T was set in the find-iso cell
delrho   = rho_m - RHO_W
ref_line = -delrho * g * w_T            # Pa (analytical Delta-sigma_zz scaling)

# Neutral plane: zero-crossing of (tau_xx - tau_zz) at x_m, between 10 and 50 km depth
diff_xm = txx[:, xm_indx] - tzz[:, xm_indx]
z_lo, z_hi = 10e3, 50e3
i_zlo = int(np.argmin(np.abs(z - z_lo)))
i_zhi = int(np.argmin(np.abs(z - z_hi)))
sc = []
for k in range(i_zlo, i_zhi):
    if diff_xm[k] * diff_xm[k+1] < 0:
        sc.append(k)
if len(sc) > 0:
    k = sc[0]
    z_np = z[k] - diff_xm[k] * (z[k+1] - z[k]) / (diff_xm[k+1] - diff_xm[k])
else:
    z_np = 30e3
z_np_km = z_np / 1e3
z_m_km  = 2 * z_np_km
z_t_km  = 80.0
gpe_est = abs(ref_line) * z_np

z_lines = [
    (z_np_km, r"$z'_{np}$"),
    (z_m_km,  r"$z'_m \sim 2\,z'_{np}$"),
    (z_t_km,  r"$z'_t$"),
]

DEPTH_LIM = (100, 0)

fig, axs = plt.subplots(2, 2, figsize=(10.5, 11.5))
ax_a, ax_b = axs[0]
ax_c, ax_d = axs[1]

T_COLOR = "#F08080"

# (a) tau_xx - tau_zz  +  Temperature twin axis
ax_a.plot((txx[:, xi_indx] - tzz[:, xi_indx]) * 1e-6, z*1e-3, "k-", lw=1.6)
ax_a.plot((txx[:, xm_indx] - tzz[:, xm_indx]) * 1e-6, z*1e-3, "k-", lw=1.6)
ax_a.plot((txx[:, tindx] - tzz[:, tindx]) * 1e-6, z*1e-3, "k-", lw=1.6)

ax_a.axvline(0, color="0.4", lw=0.5)
ax_a.set_xlim(-500, 500)
ax_a.set_ylim(*DEPTH_LIM)
ax_a.set_ylabel("Depth [km]")
ax_a.set_xlabel("in-plane differential stress\n" + r"$(\tau_{xx} - \tau_{zz})(z)$ [MPa]")
ax_aT = ax_a.twiny()
ax_aT.plot(T[:, xm_indx], z*1e-3, color=T_COLOR, lw=1.5)
ax_aT.set_xlim(0, 1500)
ax_aT.set_xlabel("Temperature [C]", color=T_COLOR)
ax_aT.tick_params(axis="x", colors=T_COLOR)

# (b) tau_xz at x_m
ax_b.plot(txz[:, xi_indx] * 1e-6, z*1e-3, "k-", lw=1.6)
ax_b.plot(txz[:, xm_indx] * 1e-6, z*1e-3, "k-", lw=1.6)
ax_b.plot(txz[:, tindx] * 1e-6, z*1e-3, "k-", lw=1.6)
ax_b.axvline(0, color="0.4", lw=0.5)
ax_b.set_xlim(-100, 100)
ax_b.set_ylim(*DEPTH_LIM)
ax_b.set_ylabel("Depth [km]")
ax_b.set_xlabel("vertical shear stress\n" + r"$\tau_{zx}(z)$ [MPa]")

# (c) Delta sigma_zz column difference
ax_c.plot(delta_sig_zz * 1e-6, z*1e-3, "k-", lw=1.6)
ax_c.fill_betweenx(z*1e-3, 0, delta_sig_zz * 1e-6,
                   where=(delta_sig_zz < 0), color="0.7", alpha=0.5)
ax_c.axvline(0, color="0.4", lw=0.5)
ax_c.axvline(ref_line * 1e-6, color="k", ls="--", lw=0.9)
ax_c.set_xlim(-200, 100)
ax_c.set_ylim(*DEPTH_LIM)
ax_c.set_ylabel("Depth [km]")
ax_c.set_xlabel("difference in vertical normal\n" + r"stresses, $\Delta\sigma_{zz}(z)$ [MPa]")
z_lab = z_m_km - 2
ax_c.plot([ref_line * 1e-6, 0], [z_lab, z_lab], "k-", lw=1.0)
ax_c.text(0.5 * ref_line * 1e-6, z_lab - 1.5, r"$\Delta P_T$",
          ha="center", va="bottom", fontsize=12)

# (d) cumulative Delta GPE*
ax_d.plot(delta_GPE_z * 1e-12, z*1e-3, "k-", lw=1.6,
          label=r"$\Delta\mathrm{GPE}^*$ (direct estimate)")
ax_d.axvline(0, color="0.4", lw=0.5)
ax_d.axvline(gpe_est * 1e-12, color="k", ls="--", lw=1.0,
             label=r"$\Delta\mathrm{GPE}^*$ (scaling estimate)")
ax_d.set_ylim(*DEPTH_LIM)
ax_d.set_ylabel("Depth [km]")
ax_d.set_xlabel("cumulative $\\Delta\\mathrm{GPE}^*$ as a\nfunction of depth [TN/m]")
ax_d.legend(loc="lower center", frameon=True, fontsize=8.5, framealpha=0.92)

panel_labels = [(ax_a, "(a)"), (ax_b, "(b)"), (ax_c, "(c)"), (ax_d, "(d)")]
for ax_, lbl in panel_labels:
    ax_.text(0.96, 0.96, lbl, transform=ax_.transAxes, ha="right", va="top",
             fontsize=14, fontweight="bold")
    for zz, _ in z_lines:
        ax_.axhline(zz, color="0.55", lw=0.55)
    xlo, xhi = ax_.get_xlim()
    for zz, lbl_z in z_lines:
        ax_.text(xlo + 0.04*(xhi - xlo), zz - 1.0, lbl_z,
                 ha="left", va="bottom", fontsize=9.5)
    ax_.spines["top"].set_visible(False)
    ax_.spines["right"].set_visible(False)

fig.suptitle(f"{MODEL_KEY} snap {TIMESTEP_INDEX}:  column profiles  "
             f"(x_T = {tloc/1e3:.0f},  x_I = {xi_loc/1e3:.0f},  x_m = {x[xm_indx]/1e3:.0f} km,  w_T = {w_T:.0f} m)",
             y=1.005)
fig.tight_layout()

figures_dir = Path("figures"); figures_dir.mkdir(exist_ok=True)
fig.savefig(figures_dir / f"profiles_{MODEL_KEY}_t{TIMESTEP_INDEX}.png",
            bbox_inches="tight", dpi=200)
plt.show()


# In[ ]:





# In[ ]:


print(w_T)


# In[ ]:


WIN_X_KM = 200
WIN_Z_KM = 110
x_lo_v = tloc - WIN_X_KM * 1e3
x_hi_v = tloc + 250e3
ix_v   = (x >= x_lo_v) & (x <= x_hi_v)

fig, ax = plt.subplots(figsize=(11, 6))
im = ax.pcolormesh(x[ix_v]*1e-3, z*1e-3, txx[:, ix_v] - tzz[:, ix_v],
                   cmap='coolwarm', shading='auto')
cbar = fig.colorbar(im, ax=ax, label='T [°C]', orientation='horizontal',
                    fraction=0.06, pad=0.13)

ax.contour(x[ix_v]*1e-3, z*1e-3, T[:, ix_v], levels=[400, 600, 800, 1000, 1200],
           colors='gold', linewidths=1.6)

#x_fit = np.linspace(x_window.min(), x_window.max(), 60)#
#ax.plot(x_fit*1e-3, (m*x_fit + b)*1e-3, 'k--', lw=1.4,
#        label=f'isotherm fit, dip = {np.rad2deg(dip):.1f}°')

# Vertical integration line
ax.plot([tloc*1e-3, tloc*1e-3], [z[0]*1e-3, z[-1]*1e-3],
        color='white', lw=2.6)
ax.plot([tloc*1e-3, tloc*1e-3], [z[0]*1e-3, z[-1]*1e-3],
        color='black', lw=1.4, label='vertical plane')

# Slab-normal path
#ax.plot(x_path*1e-3, z_path*1e-3, color='white', lw=3.0)
#ax.plot(x_path*1e-3, z_path*1e-3, color='#E5007D', lw=1.6, label='slab-normal plane')

# Trench marker
ax.plot(tloc*1e-3, 0, 'wo', mec='k', ms=9, zorder=5)
ax.annotate('trench', xy=(tloc*1e-3, 0), xytext=(tloc*1e-3, -8),
            ha='center', fontsize=11,
            arrowprops=dict(arrowstyle='-|>', lw=0.7))

ax.invert_yaxis()
ax.set_xlim(x_lo_v*1e-3, x_hi_v*1e-3)
ax.set_ylim(WIN_Z_KM, -12)
ax.set_xlabel('distance [km]')
ax.set_ylabel('depth [km]')
ax.set_aspect('equal')
ax.legend(loc='lower left', frameon=False, fontsize=10)
ax.set_title(f'{MODEL_KEY} snap {TIMESTEP_INDEX}:  trench bend, isotherm + integration paths')


# Three column lines on both panels
columns = [
    (tloc,         r'$x_T$',  '#002147'),
    (xi_loc,       r'$x_I$',  '#E5007D'),
    (x[xm_indx],   r'$x_m$',  '0.15'),
]
for x_col, label, color in columns:
    xrel = (x_col) * 1e-3
    ax.axvline(xrel, color=color, lw=1.4, alpha=0.9)
    ax.annotate(label, xy=(xrel, 0), xycoords=('data', 'axes fraction'),
                 xytext=(0, 4), textcoords='offset points',
                 ha='center', va='bottom', color=color, fontsize=11, fontweight='bold')


fig.tight_layout()




#figures_dir = Path('figures'); figures_dir.mkdir(exist_ok=True)
#fig.savefig(figures_dir / f'slab_normal_path_{MODEL_KEY}_t{TIMESTEP_INDEX}.png',
#            bbox_inches='tight', dpi=200)
plt.show()


# ## 8. Force balance — single timestep
# 
# Two views of the trailing-plate force balance at the chosen snapshot. The first omits $\mathrm{GPE}^*$ to highlight the residual; the second shows all three terms together. The dashed green curve is the residual $\Delta F_D - \Delta \mathrm{GPE}^* + F_B$ — should be small everywhere if the integrals close.
# 

# In[ ]:


#500*3300*9.8*1e-6
#col_avg(Sxx[:], tindx)


# In[ ]:


# § 8.1 — Boring form of the force balance
#
#   Delta Sigma_xx(x) + F_B(x) ~ 0
#
# In the mirrored frame (subducting plate on the right), the natural
# "right minus left" delta convention is just (current - trench):
#   Delta Sigma_xx = Sxx(x) - Sxx[tindx]
# and F_B from cumulative_trapezoid is already in the matching convention.

Sxx       = np.trapz(-p + txx, z, axis=0)        # int sigma_xx dz
delta_Sxx = Sxx - Sxx[tindx]

xkm = (x - tloc) * 1e-3

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(xkm, FB         * 1e-12, color="red", lw=2,           label=r"$F_B(x)$")
ax.plot(xkm, delta_Sxx  * 1e-12, color="b",   lw=2,           label=r"$\Delta\bar\sigma_{xx}(x)$")
ax.plot(xkm, (delta_Sxx + FB) * 1e-12,
        color="g", ls="--", lw=3, label=r"$\Delta\bar\sigma_{xx}(x) + F_B(x)$")

ax.axhline(0, color="k", lw=0.5)
ax.axvline(0, color="k", lw=0.5)
ax.axvline((xi_loc - tloc)*1e-3, color="k", lw=0.5)
ax.set_xlabel("Distance from trench [km]", fontsize=14)
ax.set_ylabel("Force per unit distance [TN/m]", fontsize=14)
ax.set_xlim(-50, 3400)
ax.set_ylim(-1.5, 1.5)
ax.legend(loc="lower right", fontsize=12, ncol=3)
fig.tight_layout()
plt.show()


# In[ ]:


# § 8.2 — Interesting form of the force balance, with topography panel
#
#   Delta F_D(x) - Delta GPE*(x) + F_B(x) ~ 0
#
# In the mirrored frame, the natural deltas are (current - trench).

Sxx = np.trapz(-p + txx, z, axis=0)     # int sigma_xx dz
Szz = np.trapz(-p + tzz, z, axis=0)     # int sigma_zz dz   (= -GPE*)

delta_Fd  = Fd  - Fd[tindx]
delta_Szz = Szz - Szz[tindx]            # = -Delta GPE*  (we plot this directly)

# Pseudo-topography from V's gradient, aligned to actual at x_I
topo_actual     = fs[0, :]
topo_pseudo     = -dVdx / (rho_m * g)
topo_pseudo_ref = topo_pseudo - topo_pseudo[xi_indx] + topo_actual[xi_indx]

xkm = (x - tloc) * 1e-3

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True,
                                gridspec_kw={"height_ratios": [1.25, 2]})

# --- Top: topography ---
ax1.plot(xkm, topo_actual,     color="k", lw=5.5, alpha=0.5,
         label="actual\nmodel\ntopo.")
ax1.plot(xkm, topo_pseudo_ref, color="k", ls="--", lw=1.5,
         label="shear stress\nsupported topo.\n" + r"$=-\frac{1}{\rho_m g}\frac{dV}{dx}$")

ax1.axhline(0, color="k", lw=0.5)
ax1.axvline(0, color="k", lw=0.5)
ax1.axvline((xi_loc - tloc)*1e-3, color="k", lw=0.5)
ax1.set_ylim(-2000, 1000)
ax1.set_ylabel("Bathymetry [m]", fontsize=14)
ax1.legend(loc="lower right", fontsize=12, ncol=2)
ax1.text((xi_loc - tloc)*1e-3 + 5, -1700,
         "first isostatic\ncolumn (" + r"$x_I$" + ")\n" + r"$dV/dx = 0$",
         fontsize=12)

# --- Bottom: ΔF_D - ΔGPE* + F_B ---
ax2.plot(xkm, FB         * 1e-12, color="red", lw=2,
         label=r"$F_B(x)$")
ax2.plot(xkm, delta_Fd   * 1e-12, color="k", ls="--", lw=2,
         label=r"$\Delta F_D(x)$")
ax2.plot(xkm, delta_Szz  * 1e-12, color="b", lw=2,
         label=r"$-\Delta\mathrm{GPE}^*(x)$")
ax2.plot(xkm, (delta_Fd + delta_Szz + FB) * 1e-12,
         color="g", ls="--", lw=3,
         label=r"$\Delta F_D - \Delta\mathrm{GPE}^* + F_B$")

ax2.axhline(0, color="k", lw=0.5)
ax2.axvline(0, color="k", lw=0.5)
ax2.axvline((xi_loc - tloc)*1e-3, color="k", lw=0.5)
ax2.set_xlabel("Distance from trench [km]", fontsize=14)
ax2.set_ylabel("Force per unit distance [TN/m]", fontsize=14)
ax2.set_xlim(-20, 3000)
ax2.set_ylim(-2.5, 3.5)
ax2.legend(loc="lower right", fontsize=12, ncol=2)

fig.tight_layout()
plt.show()


# ## 9. Diagnostics — surface topography, vertical shear, stress profiles
# 
# Internal-checking plots:
# 
# (a) surface pressure rendered as topographic-equivalent metres;
# (b) the vertical-shear integral $V(x)$;
# (c) $\partial V / \partial x$ in topographic-equivalent units;
# (d) $\tau_{xz}(z)$ and $\tau_{xx}(z)$ profiles in a column slightly outboard of the trench.
# 
# Used during development; the polished version of (a) and (c) is the comparison in §7.
# 

# In[ ]:


fig, ax = plt.subplots(figsize = (10, 5))

ax.plot( x*1e-3, (p[-1,:] - p[-1,int(tindx/2)])/(3300*9.8), color= "k", ls = '--')

#ax.set_xlim(0, tloc*1e-3 + 200)
ax.set_ylim(-250, 250)
ax.hlines(0,0, 8000)


# In[ ]:


fig, ax = plt.subplots(figsize = (10, 5))

ax.plot( x*1e-3, 1e-12*V, color= "k", ls = '--')
ax.vlines(tloc*1e-3, -10, 10, color= "k")

ax.set_ylim(-5, 5)
ax.set_xlim(0, tloc*1e-3 + 20)



# In[ ]:


fig, ax = plt.subplots(figsize = (10, 5))

ax.plot( x*1e-3, 1e-3*(dVdx/(3300*9.8)), color= "k", ls = '--')
ax.vlines(tloc*1e-3, -10, 10, color= "k")

ax.set_ylim(3, -3)
ax.set_xlim( tloc*1e-3 - 500, tloc*1e-3 + 200)


# In[ ]:


fig, ax = plt.subplots()

offset = 50
ax.plot(txz[:,tindx - offset], z*1e-3)
ax.plot(txx[:,tindx - offset], z*1e-3)
ax.hlines(1e-3*z[np.argmax(abs(txz[:,tindx - offset]))], -1e8, 1e8 )
ax.vlines(0, 0, 100, color = 'k')


ax.set_ylim( 100, 0)


# In[ ]:


np.argmax(abs(txz[:,tindx - offset]))


# In[ ]:


3300*9.8*200*100e3*1e-12


# ## 10. Time evolution
# 
# Loop over all PVTU snapshots, repeating the single-timestep pipeline (interpolate → pick trench → compute integrated quantities), and record $F_D(x_T)$ at each step. This is what drives the secular-trend plot used in the talk.
# 
# Each run saves results to `outputs/Fd_xT_<MODEL_KEY>.npz`. To produce the slide figure (which compares both models), run the loop once with `MODEL_KEY = "STD"`, then again with `MODEL_KEY = "WAL"`, then run the plot cell at the bottom of this section.
# 

# In[ ]:


import gc
from pathlib import Path

SECONDS_PER_YEAR = 31_557_600.0   # used for NormalSP::Time -> years conversion

snapshot      = []
model_time_yr = []
Fd_at_trench  = []
trench_x_m    = []

for i in range(1, len(pvtu_files)):
    ff = pvtu_files[i]

    # === Read PVTU =========================================================
    vtk_data = pv.read(ff)
    vtk_data.point_data["p"]   = vtk_data["NormalSP::Pressure"]
    vtk_data.point_data["txx"] = vtk_data["NormalSP::Stress"][:, 0]
    vtk_data.point_data["tzz"] = vtk_data["NormalSP::Stress"][:, 4]
    vtk_data.point_data["txz"] = vtk_data["NormalSP::Stress"][:, 1]
    vtk_data.point_data["T"]   = vtk_data["NormalSP::Temperature"]
    vtk_data.point_data["fs"]  = vtk_data["NormalSP::FreeSurface"]

    vel = vtk_data["NormalSP::Velocity"] / 3.17098e-10
    vtk_data.point_data["vx"] = vel[:, 0]
    vtk_data.point_data["vy"] = vel[:, 1]

    t_yr = float(np.asarray(vtk_data["NormalSP::Time"]).flat[0]) / SECONDS_PER_YEAR

    # === Grid setup ========================================================
    x_min, x_max, _, _, _, _ = vtk_data.bounds
    nx = int((x_max - x_min) / DX)
    nz = int(Z_MAX / DX)

    x = x_min + (np.arange(nx) + 0.5) * DX
    z = (np.arange(nz) + 0.5) * DX

    X, Z = np.meshgrid(x, z, indexing="xy")
    Y = Y_SURFACE - Z

    grid = pv.StructuredGrid(X.T, Y.T, np.zeros_like(X.T))

    # === Interpolate ========================================================
    interp = grid.sample(vtk_data)
    nx_pv, nz_pv, _ = interp.dimensions
    valid = interp["vtkValidPointMask"].reshape((nx_pv, nz_pv), order="F").astype(bool)

    # === Extract fields =====================================================
    p   = get_field("p")
    txx = get_field("txx")
    tzz = get_field("tzz")
    txz = get_field("txz")
    T   = get_field("T")
    vx  = get_field("vx")
    vz  = -get_field("vy")
    fs  = get_field("fs")

    # === Apply x-mirror =====================================================
    if MIRROR_X:
        p, txx, tzz, txz, T, fs, vx, vz = mirror_fields_in_x(p, txx, tzz, txz, T, fs, vx, vz)

    # === Trench pick ========================================================
    tloc, info = pick_trench_3step(
        x, z, p, vx,
        vx_depth_m=2000,
        Wp1_km=800,
        Wv_km=800,
        Wp3_km=500,
        smooth_km=30,
        subducting_side=SUBDUCTING_SIDE,
    )
    tindx = np.argmin(np.abs(x - tloc))

    # === Force calculation ==================================================
    Fd_x = np.trapz(txx - tzz, z, axis=0)

    snapshot.append(i)
    model_time_yr.append(t_yr)
    Fd_at_trench.append(1e-12 * float(Fd_x[tindx]))
    trench_x_m.append(tloc)

    # === Cleanup ============================================================
    del vtk_data, grid, interp
    del X, Y, Z, p, txx, tzz, txz, T, vx, vz, fs, Fd_x
    if i % 5 == 0:
        gc.collect()

snapshot      = np.array(snapshot, dtype=int)
model_time_yr = np.array(model_time_yr)
Fd_at_trench  = np.array(Fd_at_trench)
trench_x_m    = np.array(trench_x_m)

outputs_dir = Path("outputs")
outputs_dir.mkdir(exist_ok=True)
npz_path = outputs_dir / f"Fd_xT_{MODEL_KEY}_mirrored.npz"
np.savez(npz_path,
         snapshot=snapshot,
         model_time_yr=model_time_yr,
         Fd_at_trench=Fd_at_trench,
         trench_x_m=trench_x_m)

print(f"[{MODEL_KEY}] {len(Fd_at_trench)} snapshots  ->  {npz_path}")
print(f"           t        = {model_time_yr.min()/1e6:6.2f}  to {model_time_yr.max()/1e6:6.2f}  Myr")
print(f"           F_D(x_T) = {Fd_at_trench.min():+6.2f}  to {Fd_at_trench.max():+6.2f}  TN/m")


# ### Slide-quality plot
# 
# Loads both models from `outputs/` (re-run §9 with `MODEL_KEY = "STD"` and `MODEL_KEY = "WAL"` to populate). Saves PNG into `figures/`. To drop into the talk:
# 
# ```bash
# cp figures/Fd_xT_time_series.png ../presentation/figures/
# ```
# 
# The notebook never writes into the presentation tree directly — that's a deliberate boundary in case the analysis and the deck end up in separate repos.
# 

# In[ ]:


from pathlib import Path

outputs_dir = Path("outputs")
figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)

# Brand palette (matches presentation/theme.scss)
C_STD, C_WAL = "#002147", "#E5007D"
C_SLATE, C_RULE, C_BAND = "#404B74", "#BFC3D1", "#F2F3F5"

fig, ax = plt.subplots(figsize=(9, 5))

for key, color, label in [("STD", C_STD, "STD — standard asthenosphere"),
                          ("WAL", C_WAL, "WAL — weak asthenosphere")]:
    npz_path = outputs_dir / f"Fd_xT_{key}.npz"
    if not npz_path.exists():
        print(f"⚠  {npz_path.name} not found — re-run §9 with MODEL_KEY = {key!r}")
        continue
    d = np.load(npz_path)
    t_Myr = d["model_time_yr"] / 1e6
    ax.plot(t_Myr, d["Fd_at_trench"], "-o",
            color=color, lw=2.0, ms=5.5,
            markeredgecolor="white", markeredgewidth=0.6, label=label)

ax.axhline(0, color="k", lw=0.7)
ax.axhspan(-100, 0, color=C_BAND, zorder=-2)
ax.text(0.012, 0.97, "tension-like", transform=ax.transAxes,
        ha="left", va="top", fontsize=11, color=C_SLATE, style="italic")
ax.text(0.012, 0.03, "compression-like", transform=ax.transAxes,
        ha="left", va="bottom", fontsize=11, color=C_SLATE, style="italic")

ax.set_xlabel("Model time  [Myr]", fontsize=13)
ax.set_ylabel(r"$F_D(x_T)$   [TN m$^{-1}$]", fontsize=13)
ax.set_xlim(0, 82)
ax.set_ylim(-5.5, 5.5)
ax.legend(loc="upper right", frameon=False, fontsize=11)
ax.grid(True, alpha=0.25, color=C_RULE, lw=0.6)
ax.tick_params(labelsize=11)

fig.tight_layout()
fig.savefig(figures_dir / "Fd_xT_time_series.png", bbox_inches="tight", dpi=220)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




