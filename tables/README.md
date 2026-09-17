# tables/ — every number the paper quotes

Each `.csv` here is written by the figure script named in its `.json`
sidecar, in the same pass, from the same arrays as the figure. Nothing is
computed twice, so a table and its figure can only disagree if one of them
was not regenerated.

Read them with `scripts/tables_io.read_table(name)` — the only reader.
`value(name, model=..., quantity=...)` pulls a single number and raises if
the match is not unique. Floats are written to six significant figures so
a rerun produces no diff noise.

| table | figure | what it holds |
|---|---|---|
| `partition.csv` | `fig_partition_time` | the driving-force partition: trench pull, dynamic and static ridge push, back-tilt, the tail below z*, z* itself, the tilt fraction and the ridge:trench ratio — median and interquartile range for each |
| `nd_trench_ridge.csv` | `fig_nd_trench_ridge` | N_D at the trench and ridge, their difference, tension-like and driving fractions, and the ridge maximum as a fraction of the trench mean and maximum |
| `mechanical_thickness.csv` | `fig_mechanical_thickness` | the three thickness definitions (and two retired ones), the strength-based mean and its matching isotherm, the moment arm, and L/h for each definition |
| `force_length_scale.csv` | `fig_force_length_scale` | the length over which trench and ridge topography convert to force, and the implied surface relief |
| `bending_stress_regime.csv` | `fig_trench_stress_profiles` | τ_zx and equivalent-density peaks and depths, the τ_zx,x centre of mass (the effective moment arm), the deficit peak, the max-moment offset, the x-smoothing used and the snapshot count |
| `plate_velocities.csv` | `fig_plate_velocities` | plate, trench-migration and convergence velocities |

## Units and conventions

Forces are per unit distance along strike, in TN m⁻¹ unless the row name
says otherwise. Depths and lengths are km, stresses MPa, densities
kg m⁻³, velocities cm yr⁻¹. Positive velocities are away from the
trench in the analysis frame (subducting plate on the right).

Statistics are over model time from 8 Myr onward unless the sidecar's
`meta` says otherwise; `_median`, `_q1` and `_q3` are the run median and
quartiles. Single-snapshot quantities are at the declared mid-run
reference (36–44 Myr; see `meta.midrun_window_Myr`).

## For the manuscript

A number that is not in one of these files should not appear in the
paper. If a value is needed and has no row, the table gains a column or a
row — it is never computed in the text. The sidecars carry the provenance
(script, figure, models, and the windows/smoothing/epoch choices) so a
quoted number can be traced without reading the code.

Two notes on numbers that carry conditions:

- **the equivalent-density peak** in `bending_stress_regime.csv` is
  smoothing-dependent (`meta.x_smoothing_km`); quote it with the
  smoothing stated. The τ_zx,x **centre of mass** in the same table is
  not, and is the number the moment-arm argument rests on.
- **ridge N_D** in `nd_trench_ridge.csv` still comes from the
  notebook-built time-evolution cache, which predates the flow-based
  ridge pick (`meta.note`). Every other ridge quantity here uses the
  current pick.
