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

## Negated quantities: same colour, DASHED (Dan, 2026-09-24)

A quantity plotted as the **negative of its palette definition** keeps the
palette colour but is drawn **dashed**, and is **labelled in both
registers**. Established on `fig_budget_time` / `fig_budget_time_detrended`,
where the driving term appears as

    $\Delta\bar\sigma_{zz}\ (= -\Delta\mathrm{GPE}^{*})$

in blue dashed. The reason: the balance palette assigns blue to ΔGPE\*,
so plotting its negative solid-blue would make the colour assert the wrong
sign — and colour is read before any label. The linestyle carries what the
colour cannot.

Why this arises at all: `fig_balance_snapshot` plots **+ΔGPE\*** against
distance, while the budget-through-time figures need each curve to be its
term's contribution to the force in +x, which for the driving term is
−ΔGPE\* = +Δσ̄_zz. Writing it as Δσ̄_zz makes the balance all-additive,
ΔN_D + Δσ̄_zz + F_B = 0, and matches the decomposition already used in
the snapshot figure's fundamental-form panel (σ̄_xx = N_D + σ̄_zz).

⚠ Linestyle is also the model encoding in **encoding 3** overlays (STD
solid / WAL dashed). The two never collide in practice because encoding 3
applies only where models share an axis, and these figures put the models
in separate columns (encoding 1). A figure that needs both must say so in
its caption.

⚠ This does NOT resolve the sign-convention clash with
`fig_ridge_column_stresses` panel (c), where positive means driving
(PAPER_PLAN W28). That is a frame question, not a notation one (W31).

## Legends carry symbols, not statistics (Dan, 2026-09-24)

Trend slopes, standard deviations and other fitted numbers do **not**
belong in legend entries. They go in the figure caption, the main text, or
`tables/`. A legend should let the reader identify a curve and nothing
else. Applied to `fig_budget_time` and `fig_budget_time_detrended`, whose
legends previously carried per-curve trends and residual standard
deviations.

## Domain palette — the schematic's two domains (Dan, 2026-09-21)

The manuscript schematic colours the two additive domains of the balance,
and those colours are **carried into the data figures** so a reader meets
the same pair in both. Canonical source, not to be eyeballed or
re-picked: `trench_pull_ferrite/schematic/ridge_trench_overview_v2.tex`,
macros `cboxA` / `cboxB` (Okabe–Ito, colourblind-safe).

| domain | interval | also called | colour |
|---|---|---|---|
| non-isostatic | x_T → x_I | **trench pull domain** | `#0072B2` blue (`cboxA`) |
| isostatic | x_I → x_R | **ridge push domain** | `#D55E00` orange (`cboxB`) |

Use where a curve, band, area or label belongs to one domain — e.g.
`fig_column_anomalies`, where the trench and ridge anomaly curves take
the colour of the domain whose force their area represents.

**This palette overrides the model encoding** where the two collide: it
is only usable when the models are separated some other way (usually one
panel each), because colour is then carrying the domain, not the model.

⚠ **"Ridge push domain" names the interval, not a claim about its
content.** The force balance across it is carried by Δσ_zz, which
includes the asthenospheric pressure gradient ΔP — the tilt — as well as
the isostatic cooling signal. "Isostatic" refers to the topography being
compensated there, not to the balance being free of the tilt. Captions
must not let the name imply that the interval delivers pure ridge push;
`fig_column_anomalies` shows the tilt-removed curve precisely because it
does not.

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
- fig_column_anomalies: DOMAIN palette, not model palette (models are in
  separate panels, so colour is free to carry the domain).
- fig_nd_trench_ridge (RETIRED 2026-09-21, consolidated into
  fig_trench_resultants): black/grey/dashed within each model column
  (three related resultants of one quantity family; brand colours not
  needed in the column layout).
