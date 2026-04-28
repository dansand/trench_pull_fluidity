# Cerpa trench-pull analysis

This repository contains Jupyter notebooks that analyse a 2D Cerpa et al. Fluidity subduction model. The framework is the vertically integrated horizontal force balance ΔF_D − ΔGPE* + F_B = 0 with a focus on the role of trench topography in coupling the slab to the trailing plate.

## Conventions

- Coordinates: x rightward positive, z downward positive (z-down), z = Y_SURFACE − y.
- The model is mirrored along x at extraction so that subduction matches the MDOODZ / trench_pull_force orientation: subducting plate on the right, vx < 0 in the trailing plate, slab descending leftward. The mirror is wired through `MIRROR_X`, `SUBDUCTING_SIDE`, and `SEAWARD_SIGN` in §2 of each notebook, with a `mirror_fields_in_x` helper in §3.
- Off-diagonal stress sign flip (σ_xz_zdown = −σ_xy_yup) is applied at extraction, not buried inside V's formula.
- Diagonal stress components (σ_xx, σ_zz) are invariant under both the y→z flip and the x-mirror — no sign change, only `np.flip` for the spatial reordering.
- All stress resultants (Fd, GPE, V, FB) are *resultants*, not directional tractions. Their sign only acquires physical meaning once paired with the outward normal of a chosen plane.

## Companion repo

The math, preprint, and source-of-truth analysis for this framework live in `/Users/DSAND/projects/pyvista/trench_pull_force/`. Cross-reference that codebase when rederiving or adding diagnostics.

## Conda env

`pyvista-env` (numpy, pyvista, matplotlib, scipy, natsort).

## Outputs

Analysis writes only to `notebooks/figures/` and `notebooks/outputs/`. The presentation lives in a SEPARATE repository (`/Users/DSAND/projects/pyvista/egu_2026/`) and is populated by explicit `cp` of selected PNGs — there is no automatic pipeline. Notebooks must never write into the presentation tree.

## Layout

```
notebooks/
  cerpa_single_step.ipynb       — main force-balance analysis (single timestep + time loop)
  cerpa_slab_normal_FD.ipynb    — F_D on a slab-normal plane vs vertical
  cerpa_basal_drag.ipynb        — basal drag on horizontal plane vs along an isotherm
  subduction_schematic.ipynb    — synthetic schematic figure for the talk
  figures/                      — analysis-produced PNGs (canonical home)
  outputs/                      — npz cache files
archive/                        — superseded notebooks
papers/                         — references
scratch/                        — exploratory work, not committed/published
```

## User

Manager of the AuScope Subsurface Observatory at UniMelb; geodynamicist.
