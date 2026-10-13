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
    cell-centred grid, DX = 1 km (figure scripts and caches) / 200-500 m
    (notebooks), to Z_MAX = z_c = 80 km (deeper for slab/viscosity work;
    the column cache stores profiles to 250 km)"
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
  - "NO MIRROR (ruled 2026-09-30, ANALYSIS_CONVENTIONS §1.1). The analysis
     runs in the MODEL'S OWN ORIENTATION: subducting plate on the LEFT,
     seaward at -x, so subducting_side='left' and seaward_sign=-1 are what
     the pickers take. Until 2026-09-28 extraction set MIRROR_X = True to put
     the plate on the right and match MDOODZ; that is retired, and the suite
     no longer has a common polarity. Lengths are registered as the seaward
     distance s = x_T - x, which is identical in either orientation, so no
     published number depends on the choice.
     ⚠ The eight notebooks still call mirror_fields_in_x and are a DECLARED
     DEVIATION (conventions §9). They feed no paper figure"
  - "no water layer: rho_w = 0 in Delta-rho expressions"
  - "SIGN SENSE: the extracted tau_zx MATCHES the register's sense, pinned on
     the data at every build rather than asserted — int tau_zx dz > 0 at the
     trench (V(x_T) = +1.56 TN/m at the reference epoch). The scripts' V IS
     the register's V, V = int tau_zx dz, and dV/ds = -(rho_m - rho_w) g w
     holds for it directly on the presented axis. The former -int tau_zx dz
     code convention, and the figures that plotted -V, are retired with the
     mirror (SYMBOLOGY §7.5 divergence closed 2026-09-30).
     ⚠ The downward shear load on the trailing plate's trench-side face is
     +V, not -V: the plate lies seaward at x < x_T so that face has outward
     normal +x. It was -V until 2026-10-09, correct only while the frame was
     mirrored"

field_map:
  - {native: p,   location: points, canonical: "dynamic pressure (see native_frame.pressure)",
     transform: "sample",                              units: Pa}
  - {native: txx, location: points, canonical: "tau_xx (deviatoric)",
     transform: "sample",                              units: Pa}
  - {native: tzz, location: points, canonical: "tau_zz (deviatoric)",
     transform: "sample",                              units: Pa}
  - {native: txz, location: points, canonical: "tau_zx",
     transform: "sample; -1 (y-up flip)",              units: Pa}
  - {native: T,   location: points, canonical: "T (auxiliary; masks, isotherms)",
     transform: "sample",                              units: K}
  - {native: "NormalSP::FreeSurface", location: points,
     canonical: "free surface elevation; w = -fs (positive down)",
     transform: "BOUNDARY-NODE sample via fluidity_helpers.surface_fs — NOT
       the top row of the sampled grid. ⚠ This field is defined EVERYWHERE in
       the domain, not only on the boundary: it is the boundary elevation
       carried inward and decaying to zero within a few km, so sampling it
       below the surface returns an attenuated copy of the real topography.
       Measured: full range -2842..+1632 m at the boundary, -1695..+867 m at
       400-600 m depth (~60 % of the amplitude, pattern preserved), and
       exactly 0 below 50 km. column_profiles_cache took the grid's top row
       at z = DX/2 = 500 m until 2026-10-10; x_I moved 9 km and the trench
       pull 7.6 % when it was fixed",                units: m}
  - {native: vx,  location: points, canonical: "v_x (trench picker; auxiliary)",
     transform: "sample",                              units: m/s}
  - {native: vy,  location: points, canonical: "v_z (auxiliary)",
     transform: "sample; -1 (y-up flip)",              units: m/s}
  - {native: "SP material fraction (further_analysis)", location: points,
     canonical: "slab-top tracker (SP = 0.5 contour) and integration masks",
     transform: "sample",                              units: "-"}
```

Reference constants used in the simulations (rho = 3300 kg/m3, g = 9.8 m/s2,
alpha = 3e-5 /K nominal — NOT in the local parameter file; empirically
recovered 2026-09-02 from the in-plate dynamic-pressure gradient as 3.2e-5
(WAL) / 2.7e-5 (STD), which also pinned the density convention as
SURFACE-referenced, rho = rho0(1 - alpha(T - T_s)) with rho0 = 3300 at
T_s = 273 K; potential temperature = T_mantle = 1573 K, deep mantle
isothermal (Boussinesq, no adiabat),
T bounds, viscosity bounds, box geometry) are kept verbatim in `zenodo_materials/`; reference-value policy follows the manuscript's
`POLICY_reference_values.md`.
