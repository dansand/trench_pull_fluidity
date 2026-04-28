"""Reorganise cerpa_single_step.ipynb into labelled sections.

Reads the current notebook, rebuilds the cell list with markdown section
headings + brief descriptions, lifts model/timestep/grid config to a single
top-level cell, and drops empty/leftover cells.

Code cells that aren't being lifted into config are kept verbatim (with their
existing outputs preserved).
"""
import copy
import json
from pathlib import Path

NB_PATH = Path('/Users/DSAND/projects/pyvista/egu_2026_slabpull/notebooks/cerpa_single_step.ipynb')

nb = json.loads(NB_PATH.read_text())
orig = nb['cells']


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text}


def code(text):
    return {
        'cell_type': 'code',
        'metadata': {},
        'execution_count': None,
        'outputs': [],
        'source': text,
    }


def keep(idx, new_source=None):
    c = copy.deepcopy(orig[idx])
    c['metadata'] = {}
    if new_source is not None:
        c['source'] = new_source
        if c['cell_type'] == 'code':
            c['execution_count'] = None
            c['outputs'] = []
    return c


nc = []

# ---------------------------------------------------------------------------
# Title + intro
# ---------------------------------------------------------------------------
nc.append(md(
    "# Force-balance analysis of the Cerpa et al. Fluidity subduction model\n"
    "\n"
    "This notebook diagnoses the **vertically integrated horizontal force balance** "
    "in 2D Cerpa-style subduction simulations run with the Fluidity finite-element "
    "code. The framework is the same as in the companion repository "
    "[`trench_pull_force`](../../trench_pull_force/), where it is applied to "
    "MDOODZ7.0 outputs; here we apply it to PVTU outputs from Fluidity (STD and "
    "WAL reference models).\n"
    "\n"
    "**Working hypothesis.** Both STD and WAL models exhibit a secular shift at "
    "the trench, from near-neutral / weakly tensional early in the run to **net "
    "deviatoric compression** later. The hypothesis is that this transition "
    "reflects: plate velocity ↓ → basal drag ↓; asthenospheric pressure gradient "
    "↓ (vanishes in WAL); $\\Delta\\mathrm{GPE}^*$ ↑. With basal drag ↓ and "
    "$\\Delta\\mathrm{GPE}^*$ ↑, the only way the static force balance closes is "
    "for $F_D$ at the trench to evolve from tension-like to compression-like.\n"
    "\n"
    "**Notebook structure.**\n"
    "\n"
    "1. Setup & imports\n"
    "2. Configuration — model, timestep, grid\n"
    "3. Helper functions — field extraction and trench pickers\n"
    "4. Build the regular grid and extract fields for the chosen timestep\n"
    "5. Trench identification\n"
    "6. Vertically integrated traction quantities ($F_D$, $\\mathrm{GPE}^*$, "
    "$F_B$, $V$)\n"
    "7. Force-balance plots — single timestep\n"
    "8. Diagnostics — surface topography, vertical shear, stress profiles\n"
    "9. Time evolution — loop over timesteps\n"
    "\n"
    "**Outstanding work.** Robust through-time trench identification — the "
    "3-step picker works for individual snapshots but needs careful tuning to "
    "remain stable across the full time series.\n"
))

# ---------------------------------------------------------------------------
# 1. Setup
# ---------------------------------------------------------------------------
nc.append(md(
    "## 1. Setup & imports\n"
    "\n"
    "Numerical libraries, PyVista for VTK/PVTU I/O, and minor configuration to "
    "silence VTK chatter.\n"
))
nc.append(keep(1))

# ---------------------------------------------------------------------------
# 2. Configuration
# ---------------------------------------------------------------------------
nc.append(md(
    "## 2. Configuration\n"
    "\n"
    "Centralised configuration. Change `MODEL_KEY` to swap between reference "
    "models, and `TIMESTEP_INDEX` to load a different snapshot. Grid parameters "
    "control the regular grid the PVTU is interpolated onto for vertical "
    "integration; both the single-timestep analysis (§4) and the time loop (§9) "
    "use these same values.\n"
))
nc.append(code(
    "# === Model selection =========================================================\n"
    "DATA_ROOT = '/Users/DSAND/DATA/numerical_models/OUTPUTS/'\n"
    "\n"
    "MODELS = {\n"
    "    'STD': dict(folder='STD_RefModel', modname='subduction_with_LM'),\n"
    "    'WAL': dict(folder='WAL_RefModel', modname='subduction_with_LM'),\n"
    "}\n"
    "\n"
    "MODEL_KEY      = 'WAL'   # 'STD' or 'WAL'\n"
    "TIMESTEP_INDEX = 20      # 0-indexed; negative values count from the end (-1 = last)\n"
    "\n"
    "# === Regular-grid parameters (metres) ========================================\n"
    "DX        = 1000.0       # horizontal & vertical grid spacing on the analysis grid\n"
    "Z_MAX     = 100_000.0    # depth extent\n"
    "Y_SURFACE = 2_900_000.0  # VTK y-coordinate of the model's free surface\n"
    "\n"
    "# === Resolved paths ==========================================================\n"
    "fbase   = DATA_ROOT + MODELS[MODEL_KEY]['folder'] + '/'\n"
    "modname = MODELS[MODEL_KEY]['modname']\n"
    "\n"
    "pvtu_files = natsort.natsorted(glob.glob(fbase + modname + '*.pvtu'))\n"
    "print(f\"Model {MODEL_KEY!r}: {len(pvtu_files)} timesteps in {fbase}\")\n"
))
nc.append(md("Pick a single timestep for the analysis below."))
nc.append(code(
    "i = TIMESTEP_INDEX if TIMESTEP_INDEX >= 0 else len(pvtu_files) + TIMESTEP_INDEX\n"
    "ff = pvtu_files[i]\n"
    "print(f\"Loading timestep {i}: {ff.split('/')[-1]}\")\n"
    "vtk_data = pv.read(ff);\n"
))

# ---------------------------------------------------------------------------
# 3. Helpers
# ---------------------------------------------------------------------------
nc.append(md(
    "## 3. Helper functions\n"
    "\n"
    "- `get_field(name)` — reshapes a PyVista point-data array onto the regular "
    "grid and masks invalid samples.\n"
    "- `find_trench_x(...)` — simple trench picker: coarse pressure-minimum "
    "guess plus refined `vx` zero-crossing.\n"
    "- `pick_trench_3step(...)` — three-stage picker (pressure → velocity → "
    "directional pressure search). More robust for snapshots where the surface "
    "pressure has multiple local minima; this is the picker used by the time "
    "loop.\n"
))
nc.append(keep(6))   # field + simple find_trench_x
nc.append(keep(14))  # 3-step picker

# ---------------------------------------------------------------------------
# 4. Build grid + extract fields
# ---------------------------------------------------------------------------
nc.append(md(
    "## 4. Build grid and extract fields (single timestep)\n"
    "\n"
    "Construct a uniform cell-centred grid over `[x_min, x_max] × [0, Z_MAX]` "
    "and sample the PVTU onto it. Stress, pressure, temperature, and velocity "
    "are extracted as 2D arrays indexed `[z, x]` with depth positive downward.\n"
))
nc.append(keep(7, new_source=(
    "# === Read PVTU and stash named point-data arrays =============================\n"
    "vtk_data = pv.read(ff)\n"
    "\n"
    "vtk_data.point_data[\"p\"]   = vtk_data[\"NormalSP::Pressure\"]\n"
    "vtk_data.point_data[\"txx\"] = vtk_data[\"NormalSP::Stress\"][:, 0]\n"
    "vtk_data.point_data[\"tzz\"] = vtk_data[\"NormalSP::Stress\"][:, 4]\n"
    "vtk_data.point_data[\"txz\"] = vtk_data[\"NormalSP::Stress\"][:, 1]\n"
    "vtk_data.point_data[\"T\"]   = vtk_data[\"NormalSP::Temperature\"]\n"
    "\n"
    "vel = vtk_data[\"NormalSP::Velocity\"] / 3.17098e-10  # m/s → m/yr\n"
    "vtk_data.point_data[\"vx\"] = vel[:, 0]\n"
    "vtk_data.point_data[\"vy\"] = vel[:, 1]\n"
    "\n"
    "# === Define a regular cell-centred analysis grid =============================\n"
    "x_min, x_max, _, _, _, _ = vtk_data.bounds\n"
    "\n"
    "nx = int((x_max - x_min) / DX)\n"
    "nz = int(Z_MAX / DX)\n"
    "\n"
    "x = x_min + (np.arange(nx) + 0.5) * DX     # metres\n"
    "z = (np.arange(nz) + 0.5) * DX              # depth, metres (positive down)\n"
    "\n"
    "X, Z = np.meshgrid(x, z, indexing=\"xy\")\n"
    "Y    = Y_SURFACE - Z                        # depth → VTK y\n"
    "\n"
    "grid = pv.StructuredGrid(X.T, Y.T, np.zeros_like(X.T))\n"
    "\n"
    "# === Interpolate VTK fields onto the regular grid ============================\n"
    "interp = grid.sample(vtk_data)\n"
    "\n"
    "nx_pv, nz_pv, _ = interp.dimensions\n"
    "valid = interp[\"vtkValidPointMask\"].reshape((nx_pv, nz_pv), order=\"F\").astype(bool)\n"
    "\n"
    "# === Final canonical-form arrays =============================================\n"
    "# field[z_index, x_index] lives at (x[x_index], z[z_index])\n"
    "p   = get_field(\"p\")\n"
    "txx = get_field(\"txx\")\n"
    "tzz = get_field(\"tzz\")\n"
    "txz = get_field(\"txz\")\n"
    "T   = get_field(\"T\")\n"
    "vx  = get_field(\"vx\")\n"
    "vz  = -get_field(\"vy\")            # positive downward\n"
    "\n"
    "print(\"x (m):\", x[0], \"→\", x[-1], \"  nx =\", len(x))\n"
    "print(\"z (m):\", z[0], \"→\", z[-1], \"  nz =\", len(z))\n"
    "print(\"p.shape =\", p.shape)\n"
)))

# ---------------------------------------------------------------------------
# 5. Trench identification
# ---------------------------------------------------------------------------
nc.append(md(
    "## 5. Trench identification\n"
    "\n"
    "Two pickers are exercised here. The simple one (`find_trench_x`) gives a "
    "coarse anchor; the 3-step picker (`pick_trench_3step`) refines it via a "
    "directional pressure search on the subducting-plate side.\n"
    "\n"
    "*This remains the most fragile step in the workflow — see the visual check "
    "at the end of the section if a pick looks wrong.*\n"
))
nc.append(keep(8))   # surface_p
nc.append(keep(9))   # find_trench_x first call
nc.append(keep(10))  # tindx
nc.append(keep(15))  # 3-step picker
nc.append(keep(17))  # visual check

# ---------------------------------------------------------------------------
# 6. Integrated traction quantities
# ---------------------------------------------------------------------------
nc.append(md(
    "## 6. Vertically integrated traction quantities\n"
    "\n"
    "Compute the depth-integrated quantities that enter the static horizontal "
    "force balance:\n"
    "\n"
    "$$\\Delta F_D - \\Delta \\mathrm{GPE}^* + F_B = 0$$\n"
    "\n"
    "- $F_D = \\overline{\\tau_{xx} - \\tau_{zz}}$ — in-plane differential "
    "stress resultant (tension-like / compression-like).\n"
    "- $\\mathrm{GPE}^* = \\overline{p - \\tau_{zz}} = -\\overline{\\sigma_{zz}}$ "
    "— pressure-component, carries topographic information including the "
    "trench-pull contribution. The constant offset is chosen so "
    "$\\mathrm{GPE}^*(x_T) = F_D(x_T)$, which makes the curves directly "
    "comparable in §7.\n"
    "- $V = -\\overline{\\tau_{xz}}$ and $\\partial V / \\partial x$ — vertical-"
    "shear-stress integral; $\\partial V / \\partial x$ in topographic-equivalent "
    "units (metres) is plotted in §8.\n"
    "- $F_B$ — cumulative basal traction (referenced to zero at the trench).\n"
))
nc.append(keep(19))  # import cumulative_trapezoid
nc.append(keep(20))  # Fd, GPE_, GPE, V, dVdx, tau_b, FB

# ---------------------------------------------------------------------------
# 7. Force balance plots
# ---------------------------------------------------------------------------
nc.append(md(
    "## 7. Force balance — single timestep\n"
    "\n"
    "Two views of the trailing-plate force balance at the chosen snapshot. The "
    "first omits $\\mathrm{GPE}^*$ to highlight the residual; the second shows "
    "all three terms together. The dashed green curve is the residual "
    "$\\Delta F_D - \\Delta \\mathrm{GPE}^* + F_B$ — should be small everywhere "
    "if the integrals close.\n"
))
nc.append(keep(26))  # FB + Fd, no GPE
nc.append(keep(27))  # full force balance

# ---------------------------------------------------------------------------
# 8. Diagnostics
# ---------------------------------------------------------------------------
nc.append(md(
    "## 8. Diagnostics — surface topography, vertical shear, stress profiles\n"
    "\n"
    "Internal-checking plots:\n"
    "\n"
    "(a) surface pressure rendered as topographic-equivalent metres;\n"
    "(b) the vertical-shear integral $V(x)$;\n"
    "(c) $\\partial V / \\partial x$ in topographic-equivalent units;\n"
    "(d) $\\tau_{xz}(z)$ and $\\tau_{xx}(z)$ profiles in a column slightly "
    "outboard of the trench.\n"
    "\n"
    "Used during development; not all of these need to make it into a "
    "publishable figure set.\n"
))
nc.append(keep(11))  # surface topography
nc.append(keep(23))  # V plot
nc.append(keep(25))  # dVdx
nc.append(keep(28))  # txz/txx profile
nc.append(keep(29))  # argmax of |txz|
nc.append(keep(12))  # magnitude estimate

# ---------------------------------------------------------------------------
# 9. Time evolution
# ---------------------------------------------------------------------------
nc.append(md(
    "## 9. Time evolution\n"
    "\n"
    "Loop over all PVTU snapshots, repeating the single-timestep pipeline "
    "(interpolate → pick trench → compute integrated quantities), and record "
    "$F_D(x_T)$ at each step. This is what drives the secular-trend plot used "
    "in the talk.\n"
    "\n"
    "*To stash a model's time series for cross-model comparison after the loop "
    "completes, uncomment the relevant line in the cell below the loop.*\n"
))
nc.append(keep(31, new_source=(
    "import gc\n"
    "\n"
    "Fd_tx = []\n"
    "\n"
    "for i in range(1, len(pvtu_files)):\n"
    "    ff = pvtu_files[i]\n"
    "\n"
    "    # === Read PVTU =========================================================\n"
    "    vtk_data = pv.read(ff)\n"
    "    vtk_data.point_data[\"p\"]   = vtk_data[\"NormalSP::Pressure\"]\n"
    "    vtk_data.point_data[\"txx\"] = vtk_data[\"NormalSP::Stress\"][:, 0]\n"
    "    vtk_data.point_data[\"tzz\"] = vtk_data[\"NormalSP::Stress\"][:, 4]\n"
    "    vtk_data.point_data[\"txz\"] = vtk_data[\"NormalSP::Stress\"][:, 1]\n"
    "\n"
    "    vel = vtk_data[\"NormalSP::Velocity\"] / 3.17098e-10\n"
    "    vtk_data.point_data[\"vx\"] = vel[:, 0]\n"
    "    vtk_data.point_data[\"vy\"] = vel[:, 1]\n"
    "\n"
    "    # === Grid setup (uses module-level config: DX, Z_MAX, Y_SURFACE) ========\n"
    "    x_min, x_max, _, _, _, _ = vtk_data.bounds\n"
    "    nx = int((x_max - x_min) / DX)\n"
    "    nz = int(Z_MAX / DX)\n"
    "\n"
    "    x = x_min + (np.arange(nx) + 0.5) * DX\n"
    "    z = (np.arange(nz) + 0.5) * DX\n"
    "\n"
    "    X, Z = np.meshgrid(x, z, indexing=\"xy\")\n"
    "    Y = Y_SURFACE - Z\n"
    "\n"
    "    grid = pv.StructuredGrid(X.T, Y.T, np.zeros_like(X.T))\n"
    "\n"
    "    # === Interpolate ========================================================\n"
    "    interp = grid.sample(vtk_data)\n"
    "    nx_pv, nz_pv, _ = interp.dimensions\n"
    "    valid = interp[\"vtkValidPointMask\"].reshape((nx_pv, nz_pv), order=\"F\").astype(bool)\n"
    "\n"
    "    # === Extract fields =====================================================\n"
    "    p   = get_field(\"p\")\n"
    "    txx = get_field(\"txx\")\n"
    "    tzz = get_field(\"tzz\")\n"
    "    vx  = get_field(\"vx\")\n"
    "\n"
    "    # === Trench pick ========================================================\n"
    "    tloc, info = pick_trench_3step(\n"
    "        x, z, p, vx,\n"
    "        vx_depth_m=2000,\n"
    "        Wp1_km=800,\n"
    "        Wv_km=800,\n"
    "        Wp3_km=500,\n"
    "        smooth_km=30,\n"
    "        subducting_side=\"left\",\n"
    "    )\n"
    "    tindx = np.argmin(np.abs(x - tloc))\n"
    "\n"
    "    # === Force calculation ==================================================\n"
    "    Fd = np.trapz(txx - tzz, z, axis=0)\n"
    "    Fd_tx.append(1e-12 * Fd[tindx])\n"
    "\n"
    "    # === Cleanup ============================================================\n"
    "    del vtk_data, grid, interp\n"
    "    del X, Y, Z, p, txx, tzz, vx, Fd\n"
    "\n"
    "    if i % 5 == 0:\n"
    "        gc.collect()\n"
    "\n"
    "Fd_tx = np.array(Fd_tx)\n"
    "print(f\"Collected F_D(x_T) for {len(Fd_tx)} timesteps\")\n"
)))

nc.append(md(
    "**Stash this run's time series** for cross-model comparison "
    "(uncomment the line that matches `MODEL_KEY`):\n"
))
nc.append(code(
    "# Fd_tx_std = np.copy(Fd_tx)\n"
    "# Fd_tx_wal = np.copy(Fd_tx)\n"
))

nc.append(md("Trench differential-stress resultant through time:"))
nc.append(keep(35))

nb['cells'] = nc

NB_PATH.write_text(json.dumps(nb, indent=1) + '\n')
print(f"Wrote {len(nc)} cells to {NB_PATH}")
