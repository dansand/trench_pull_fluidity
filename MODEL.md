# MODEL.md — Cerpa et al. Fluidity subduction models (STD / WAL)

Structured model description and raw-field disambiguation for this repo's analysis.
Schema: `trench-pull-model/1` (defined in the umbrella project,
`tools/MODEL_SCHEMA.md`); symbols follow the trench-pull umbrella project's `SYMBOLOGY.md`.

```yaml
schema: trench-pull-model/1

model:
  key: STD | WAL
  code: Fluidity (finite element, unstructured, adaptive; Stokes, composite rheology)
  provenance: >
    Cerpa, Sigloch, Garel, Heuret, Davies, Mihalynuk — "The effect of a weak
    asthenospheric layer on surface kinematics, subduction dynamics and slab
    morphology in the lower mantle", JGR Solid Earth. Data archive:
    https://zenodo.org/records/6817177. Original input and parameter files kept
    verbatim in zenodo_materials/.
  cases:
    - STD_RefModel — standard reference case
    - WAL_RefModel — weak-asthenospheric-layer case (viscosity factor 0.5,
      0-220 km depth, T_LAB = 1373 K)

domain:
  extents:
    x: "[0, 8000] km native; analysis grids typically span the plate region"
    z: "[0, 2900] km depth (native y-up, Y_SURFACE = 2.9e6 m)"
  resolution: "unstructured adaptive mesh; analysis samples onto a uniform
    cell-centred grid, DX = 200 m (main notebooks) / 500 m (further_analysis),
    to Z_MAX = 75 km (deeper for slab/viscosity notebooks)"
  key_locations:
    initial_trench: "x = 4000 km; subducting plate x in [0, 4000] km (native)"
    lower_mantle_interface: "z = 660 km"
    initial_slab: "dip ~12.6 deg (pi/14.3), bending radius 250 km"
  time: "~80 Myr, 41 PVTU snapshots per case"

native_frame:
  coordinates: "y positive UP; analysis depth z = Y_SURFACE - y"
  stress_sign: "engineering — tension positive, compression negative"
  stress_register: "txx, tzz, txz are DEVIATORIC stress components"
  pressure: "DYNAMIC pressure (lithostatic subtracted). Single-column absolute
    integrals are therefore offset, but every between-column DIFFERENCE is
    directly comparable to full-pressure results — the lithostatic part cancels"
  layout: "point data on the unstructured mesh; sampled via
    pyvista StructuredGrid.sample, invalid points masked (vtkValidPointMask)"

extraction:
  - "sample PVTU point data onto the uniform grid; reshape [z, x], NaN-mask"
  - "y-up to z-down: tau_zx = -txz at extraction (sigma_zx^z-down =
     -sigma_xy^y-up); diagonal components invariant"
  - "MIRROR_X = True: flip along x so the subducting plate sits on the RIGHT
     (matches the MDOODZ analysis frame); mirror flips signs of vx and the
     off-diagonal stress — bundled in mirror_fields_in_x"
  - "no water layer: rho_w = 0 in Delta-rho expressions"

field_map:
  - {native: p,   location: points, canonical: "dynamic pressure (see native_frame.pressure)",
     transform: "sample; mirror",                              units: Pa}
  - {native: txx, location: points, canonical: "tau_xx (deviatoric)",
     transform: "sample; mirror",                              units: Pa}
  - {native: tzz, location: points, canonical: "tau_zz (deviatoric)",
     transform: "sample; mirror",                              units: Pa}
  - {native: txz, location: points, canonical: "tau_zx",
     transform: "sample; -1 (y-up flip); mirror sign flip",    units: Pa}
  - {native: T,   location: points, canonical: "T (auxiliary; masks, isotherms)",
     transform: "sample; mirror",                              units: K}
  - {native: fs,  location: points, canonical: "free surface; w = -fs (positive down)",
     transform: "top-row sample; mirror",                      units: m}
  - {native: vx,  location: points, canonical: "v_x (trench picker; auxiliary)",
     transform: "sample; mirror sign flip",                    units: m/s}
  - {native: vy,  location: points, canonical: "v_z (auxiliary)",
     transform: "sample; mirror",                              units: m/s}
  - {native: "SP material fraction (further_analysis)", location: points,
     canonical: "slab-top tracker (SP = 0.5 contour) and integration masks",
     transform: "sample; mirror",                              units: "-"}
```

Reference constants used in the simulations (rho = 3300 kg/m3, g = 9.8 m/s2,
T bounds, viscosity bounds, box geometry) are kept verbatim in `zenodo_materials/`; reference-value policy follows the manuscript's
`POLICY_reference_values.md`.
