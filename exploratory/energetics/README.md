# exploratory/ — NOT PART OF THE PAPER

Everything under this directory is proof-of-concept work that has **not**
been adopted into the manuscript, and must not leak into it.

**The containment rule.** Scripts here write their figures and tables
*beside themselves*, never into the repository's top-level `figures/` or
`tables/`. Those two directories are mirrored into the manuscript tree
(`gpe_geodynamica/jgr_solid_earth/`) and must continue to contain exactly
the paper's figure set and nothing else — that is the property the whole
provenance chain rests on.

Nothing here is covered by the figure/table conventions, the audit, or the
`fig_<name>.py -> figures/fig_<name>.png` naming rule.

If a piece of this is ever adopted, it moves: renamed to `fig_*`, placed in
`scripts/`, given a CSV and JSON sidecar, and entered in the audit ledger.
Until Dan decides that, it stays here.

## energetics/ (2026-10-12)

Does the rate of gravitational potential energy release balance viscous
dissipation, and how is the dissipation distributed? Proof of concept at
the bundled reference snapshot only.
