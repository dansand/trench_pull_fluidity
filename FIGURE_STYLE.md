# Figure style register — STD/WAL symbolisation and shared conventions

Started 2026-09-15 (Dan: "keep a log or mapping for how we symbolise the
STD versus WAL models — not every figure admits the same use of line
style or colour, but where possible some homogeneity would be useful").
Every `fig_` script follows this register or declares its deviation here.
The notebooks' developed figures are the starting point for all styling
(never reinvent); this file records the conventions they embody.

## Model encoding (STD vs WAL)

1. **Preferred: side-by-side columns, STD left | WAL right** (no
   per-model line encoding needed). Used by: fig_headline_tracking,
   fig_column_anomalies (+ normalised), fig_ridge_density_check,
   fig_balance_snapshot, fig_nd_trench_ridge.
2. **Overlaid on one axis: model = BRAND COLOUR** — STD navy `#002147`,
   WAL magenta `#E5007D` (the talk/poster palette; established in the
   time-evolution notebook's series figures). Quantity is then encoded
   by linestyle/marker: primary solid `-o`, comparison dashed `--s`,
   white marker edges (`markeredgecolor='white'`, width 0.6).
3. Overlays where colour must encode the QUANTITY instead (the balance
   palette, below): model falls back to linestyle — STD solid, WAL
   dashed — and the figure declares it in the caption.

## Quantity (term) palette — the balance family

From the single-step notebook / poster brand table: N_D **black** ·
ΔGPE*/Δσ̄_xx **blue** · F_B **red** · closure/sum **green dashed** ·
trench column orange · ridge column purple. Direction glyph
("interpreting a change") uses the same colours.

## Other shared conventions

- Axis labels per SYMBOLOGY §3: `Force per unit distance [TN/m]`,
  `Depth [km]`, `Distance from trench [km]`, `$w$ [m] (positive
  downward)`. (The time-evolution notebook's "force per unit length
  [TN m⁻¹]" is a legacy label — swept to the canonical form in fig_
  scripts.) Moment: `$M_T$ [$10^{17}$ N]` (bending-notebook convention).
- Time axes: `Model time [Myr]`, model time always (never step index).
- Grid on time-series figures: `alpha=0.25, color='#BFC3D1', lw=0.6`.
- Zero lines: `axhline(0, color='k', lw≈0.5–0.7)`.
- Seaward side only on plate-profile figures (xlim from ≈ −50 km).
- Topography: display smoothing seaward of x_I only; the trench is raw
  (real narrow structure — never filter it away).
- Axis limits data-driven from the plotted curves (~10 % pad), shared
  across model columns.

## Declared exceptions

- **fig_headline_tracking: greyscale only** (Dan's ruling was specific
  to this figure, not blanket).
- fig_nd_trench_ridge: black/grey/dashed within each model column
  (three related resultants of one quantity family; brand colours not
  needed in the column layout).
