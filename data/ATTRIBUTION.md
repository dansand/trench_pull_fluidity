# Bundled model output — provenance and licence

The two snapshots in this directory are **not ours**. They are redistributed
from the data archive accompanying Cerpa et al. (2022).

## What is here

| | |
|---|---|
| `STD_RefModel/subduction_with_LM_20*` | the standard case at t = 40 Myr |
| `WAL_RefModel/subduction_with_LM_20*` | the weak-asthenospheric-layer case at the same epoch |

Two snapshots of the 41 in each run — the declared reference epoch of this
study. They are included so that `notebooks/01_single_timestep.ipynb` and the
main-text force-balance figure run without a download. Nothing has been
modified: the `.pvtu` files and their pieces are byte-for-byte as published.

## Source

Cerpa, N., Sigloch, K., Garel, F., Heuret, A., Davies, D. R., and Mihalynuk, M.
(2022). *Data for "The effect of a weak asthenospheric layer on surface
kinematics, subduction dynamics and slab morphology in the lower mantle"*,
Journal of Geophysical Research: Solid Earth. Zenodo.
https://doi.org/10.5281/zenodo.6817177

The paper the archive accompanies:
https://doi.org/10.1029/2022JB024494

## Licence

The archive is published under **Creative Commons Attribution 4.0 International
(CC-BY-4.0)**, which permits redistribution of the material, including in
modified form, provided attribution is given. That is the basis on which these
two snapshots appear here, and this file is the attribution.

**If you use these data for anything, cite Cerpa et al. (2022) and the Zenodo
record** — not this repository. This repository is post-processing; the model
output is theirs.

The full archive — 41 snapshots per run, plus the Fluidity input file, the
parameter file and the mesh needed to reproduce the simulations — is at the DOI
above. `../zenodo_materials/` holds the archive's own README and parameter file
verbatim.
