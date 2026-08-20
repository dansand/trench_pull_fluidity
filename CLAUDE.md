# Cerpa trench-pull analysis

This repository contains Jupyter notebooks that analyse a 2D Cerpa et al. Fluidity subduction model. The framework is the vertically integrated horizontal force balance ΔF_D − ΔGPE* + F_B = 0 with a focus on the role of trench topography in coupling the slab to the trailing plate.

## Conventions

- Coordinates: x rightward positive, z downward positive (z-down), z = Y_SURFACE − y.
- The model is mirrored along x at extraction so that subduction matches the MDOODZ / trench_pull_force orientation: subducting plate on the right, vx < 0 in the trailing plate, slab descending leftward. The mirror is wired through `MIRROR_X`, `SUBDUCTING_SIDE`, and `SEAWARD_SIGN` in §2 of each notebook, with a `mirror_fields_in_x` helper in §3.
- Off-diagonal stress sign flip (σ_xz_zdown = −σ_xy_yup) is applied at extraction, not buried inside V's formula.
- Diagonal stress components (σ_xx, σ_zz) are invariant under both the y→z flip and the x-mirror — no sign change, only `np.flip` for the spatial reordering.
- All stress resultants (Fd, GPE, V, FB) are *resultants*, not directional tractions. Their sign only acquires physical meaning once paired with the outward normal of a chosen plane.

## Companion repo

The math, preprint, and source-of-truth analysis for this framework live in `/Users/DSAND/projects/pyvista/trench_pull/trench_pull_force/`. Cross-reference that codebase when rederiving or adding diagnostics.

## Source data

Cerpa Fluidity outputs are from: Cerpa, N. G., Sigloch, K., Garel, F., Heuret, A., Davies, D. R., Mihalynuk, M. *The effect of a weak asthenospheric layer on surface kinematics, subduction dynamics and slab morphology in the lower mantle.* JGR Solid Earth. Data archive: <https://zenodo.org/records/6817177>. The original Fluidity input file, parameter file (`constants_weakasth2.py`), and README from the archive are kept verbatim in `zenodo_materials/`.

- `STD_RefModel` — standard case. `WAL_RefModel` — weak-asthenospheric-layer case (viscosity factor 0.5, depth range 0–220 km, T_LAB = 1373 K).
- **Box geometry:** 8000 km wide × 2900 km deep. Initial trench at x = 4000 km; subducting plate x ∈ [0, 4000] km, overriding plate x ∈ [4000, 8000] km. Lower-mantle interface at z = 660 km. Initial slab dip α = π/14.3 ≈ 12.6°, bending radius R = 250 km.
- **Reference constants used in the simulation:** ρ = 3300 kg/m³, g = 9.8 m/s², T_surface = 273 K, T_mantle = 1573 K (1300 °C), viscosity bounds 10¹⁸–10²⁵ Pa·s. Composite rheology: Byerlee + diffusion + dislocation + Peierls creep.

## Conda env

`pyvista-env` (numpy, pyvista, matplotlib, scipy, natsort, jupyterlab). Reproducible via the bundled [`environment.yml`](environment.yml) at repo root. After `conda env create`, a `python -m ipykernel install --user --name pyvista-env` step is required to register the env as a Jupyter kernel — without it, notebooks fall back silently to the base anaconda kernel.

Two compatibility notes:
- All four notebooks include `if not hasattr(np, 'trapz'): np.trapz = np.trapezoid` after `import numpy as np`. This is a forward-compat shim for numpy ≥ 2.0 (which removed `np.trapz` in favour of `np.trapezoid`). Original env had numpy 1.26 + scipy 1.13 where `np.trapz` and `scipy.integrate.cumtrapz` still existed; the shim keeps the notebooks running on both old and new env.
- `cerpa_helpers.py` is the shared helpers module imported by all notebooks. Despite the notebook rename (`cerpa_*.ipynb` → `fluidity_*.ipynb`), the helpers module kept its original name — renaming it would have meant updating import lines in every notebook for no real benefit.

## Outputs

Analysis writes only to `notebooks/figures/` and `notebooks/outputs/`. The presentation lives in a SEPARATE repository (`/Users/DSAND/projects/pyvista/trench_pull/egu_2026/`) and is populated by explicit `cp` of selected PNGs — there is no automatic pipeline. Notebooks must never write into the presentation tree.

## Layout

```
notebooks/
  fluidity_single_step.ipynb               — main force-balance analysis (single timestep + time loop)
  fluidity_time_evolution.ipynb            — time-evolution analysis with npz cache
  cerpa_helpers.py                         — shared helpers (importable from any notebook)
  further_analysis/
    fluidity_slab_normal_FD.ipynb          — F_D on a slab-normal plane vs vertical (+ §12 slab-top FD wip)
    fluidity_basal_drag.ipynb              — basal drag on horizontal plane vs along an isotherm
    fluidity_viscosity_evolution.ipynb     — 6-panel slab viscosity figure (Cerpa-style)
  figures/                                 — analysis-produced PNGs (canonical home)
  outputs/                                 — npz cache files
zenodo_materials/                          — original Cerpa parameter file + README from the data archive
```

## Down-dip force balance — shelved on the `downdip-force-balance` branch

§12 of `fluidity_slab_normal_FD.ipynb` ("$F_D$ along the slab-top SP boundary") — the multi-plane slab-top F_D diagnostic with SP-fraction masking and a depth-decoupled transition to T-only masking at ~125 km — is finished as committed and **stays on `main`** (decided 2026-08-20; see `BRANCHING_PLAN.md`).

The open investigation built on top of it — cumulative down-dip buoyancy (W_cum), the analytical interface-shear term (F_int), the unbent-slab visualisation, and the roadmap (cell 39 of the notebook) — lives only on the `downdip-force-balance` branch. That branch rewrites §12's central pipeline cell, so it will conflict with `main` inside §12: treat it as a reference to read from, not a branch to merge back blind. The repository is the source of truth for all of this — there are no external roadmap or script files.

Key methodological decisions already made on this thread (don't re-litigate when restarting):
- Slab geometry tracker is the SP=0.5 contour, not an isotherm (isotherms have two-branch ambiguity at depth).
- Walk the contour in both directions from a seed at `(tloc, 5 km)`; no synthetic z=0 extension.
- Gaussian-filter the polyline (sigma ~ 15–20 pts) before fitting the spline.
- Slab-normal lines are bilateral (`SLAB_NORMAL_UP_KM ≈ 10–20`, `SLAB_NORMAL_DOWN_KM ≈ 120`).
- Mask: `SP > 0.5 & T < T_MAX_K` above `DECOUPLING_DEPTH_KM` (≈125 km), `T < T_MAX_K` only below — handles the thermal halo that develops on top of the deep slab.
- Reference density for buoyancy: lowest unmasked density on each slab-normal line (adapts with depth).
- Analytical interface shear: constant τ ≈ 10 MPa, applied down-dip of trench, capped at decoupling depth.
- Curvature coupling for the slab-parallel force balance: $V \cdot \Delta\theta$ summed between consecutive eval points (no explicit κ computation needed); $V = \int\sigma_{\xi\eta}\,d\eta$ is the *transverse-stress resultant*, the down-dip analog of $V(x) = \int\sigma_{xz}\,dz$ from the trailing-plate balance.

