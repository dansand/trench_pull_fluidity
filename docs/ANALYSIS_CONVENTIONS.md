# Analysis conventions — the suite's methods register

**Status:** ACTIVE 2026-08-31 — §2 (references/closure pinning), §3.1 (fore-arc rule), §4.2 (±25 km standard) and §4b (reference snapshots) ruled by Dan; remainder as drafted. The methods sibling of
`SYMBOLOGY.md`: where that file governs *how a quantity is written*, this one
governs *how it is measured*. Scope: the force-balance analysis conventions
shared by the model repos (`trench_pull_mdoodz`, `trench_pull_fluidity`,
`trench_pull_aspect`, and future members). Not in scope: notation
(`SYMBOLOGY.md`), reference values (`POLICY_reference_values.md`), raw-field
disambiguation (each repo's `MODEL.md` — this file assumes those).

**Canonical home:** the umbrella root. The repos carry no copy; each notebook
declares its instantiation of the rules below, and deviations are declared,
never silent. The goal, per Dan (2026-08-31): the repos are presentable as a
**set of reproducible analyses whose consistency — and whose declared
differences — can be audited from one document.**

**Form of each convention:** a *rule* (suite-wide), its *per-repo
instantiation* (may differ for model-specific reasons), and any *declared
deviation* with its justification. A convention with an undeclared deviation
is a bug.

---

## 1. Extraction discipline

1. **`MODEL.md` first.** Every repo specifies the raw-field → canonical
   mapping (frame, sign register, pressure meaning, ordered transforms)
   before any notebook exists. The analysis frame is $x$ rightward, $z$
   downward positive, engineering (tension-positive) signs.

   **THE SUBDUCTING-PLATE SIDE IS NO LONGER SUITE-WIDE (2026-09-30, §10
   procedure).** It was "subducting plate on the right", which fluidity
   achieved by mirroring its output. Fluidity now runs in the **model's own
   orientation**: subducting plate on the LEFT, seaward at $-x$. MDOODZ and
   ASPECT are unchanged and keep the plate on the right. Each repo's
   `MODEL.md` states its own polarity; the pickers take it as
   `subducting_side` / `seaward_sign`.

   *Why.* Consistency with the source publication, and in the native frame
   plate velocity, $V$ and the trench-facing outward normal are all
   positive.

   *What it does and does not change.* Column integrals of $\sigma_{xx}$
   and $\sigma_{zz}$ are invariant under $x\to-x$, so **no force number
   changes**: $N_D$, $\mathrm{GPE}^*$, $\Delta N_D$, $\Delta\mathrm{GPE}^*$,
   $F_B$ and $M$ are identical to ~$10^{-15}$; $\tau_{zx}$ and $v_x$ negate;
   column positions reflect. Verified over all 37 snapshots of both runs —
   26 of 26 checks. **Re-run that reflection test after any future frame
   change**: it caught four directional bugs that code review did not,
   including one that made a headline figure's residuals look six times
   better than they are.

   *Registration.* Lengths are measured as the **seaward distance
   $s = x_T - x$**, positive toward the ridge, zero at the trench. This is
   frame-independent — in the mirrored frame the same quantity is
   $x - x_T$ — so every length in the manuscript survives any future frame
   decision. The symbol is provisional: $s$ is also the pending upstream
   proposal for the slab-frame down-dip coordinate (`SYMBOLOGY.md` §7.1),
   and the collision must be resolved upstream before either is used in
   print.

   **THE CONVENTION, ruled by Dan 2026-09-30: compute native, display
   mirrored.** Everything — caches, notebooks, figure scripts — runs in
   the model's native orientation. Any figure that has a physical
   left–right (a cross-section, the balance schematic) is *drawn* with
   the trench at the left and seaward increasing rightward, which is the
   reflection of the native frame. That flip happens at plot time, in
   the plotting script, and nowhere else.

   **NOTHING NATIVE REACHES THE READER (Dan, 2026-10-09).** The
   display rule above was written as if it governed layout only. It
   does not. **Every presented quantity — every plotted curve, every
   tabulated number, every sign in a caption, every outward normal an
   argument appeals to — is the value in the frame the reader is
   looking at.** The native frame is bookkeeping and stops at the
   figure script.

   *This is enforceable because the presented axis is $s = x_T - x$.*
   Measured on the data at the reference epoch: $V(x_T) = +1.56$ TN/m
   and $dV/ds = -\rho g w$, both as the register defines them. $F_B$ is
   checked by the closure, which holds to 1–2 % of the terms. So the
   presented frame already satisfies the register; what has to be
   guarded is that a script does not quietly hand through a native
   value.

   ⚠ It did once. `fig_trench_resultants` plotted $-V$ and called it
   the downward shear load on the trailing plate's trench face. That
   was right while the plate sat at $x > x_T$ and the face's outward
   normal was $-x$ — negative normal times the then-negative $V$ gives
   a positive, downward load. The frame changed on 2026-09-28, the
   sign did not follow, and from then until 2026-10-09 a main-text
   figure showed $-1.60$ TN/m under a label reading "downward". The
   script printed the plotted value and the extracted $V$ on the same
   line and neither was questioned. **Any future frame or sign change
   must re-check every presented quantity, not only the fields** — the
   reflection test (above) covers the fields and did not catch this.

   *Why this is not two frames.* No figure in the paper plots the model's
   $x$ coordinate. Every horizontal axis is the seaward distance $s$,
   model time, or a stress/force quantity, and $s$ is identical in both
   orientations. The frame is therefore internal bookkeeping: it fixes
   the signs of $\tau_{zx}$, $v_x$, $V$ and $F_B$ and nothing a reader
   sees. Display orientation is a separate, purely presentational
   choice, and pinning it to trench-left costs one reflection in the one
   or two scripts that draw a cross-section.

   ⚠ **Correction to the record (2026-09-30).** The change was argued
   here as making "the driving force positive by definition". It does
   not. Whether a plotted force term reads positive-driving or
   positive-resisting follows from how the balance is *arranged*
   ($\Delta\mathrm{GPE}^* = \Delta N_D + F_B$ puts the driving term and
   both resisting terms all positive), not from the frame. What the
   frame actually bought: $V$ and $v_x$ carry the sense the register
   defines, retiring the `SYMBOLOGY.md` §7.5 divergence, and the raw
   fields match the source publication.

   ⚠ **Mixed-frame status.** The eight fluidity notebooks still call
   `mirror_fields_in_x`. They feed no paper figure, but the convention
   above is single-frame, so they are a declared deviation until
   converted (§9).
2. **Step-0 empirical pinning.** Every notebook re-verifies the dangerous
   ambiguities on the data before quoting a result: pressure meaning
   ($dp/dz$ vs $\rho g$), register checks, and the **$\tau_{zx}$ sign
   ritual** — closure computed for both signs of the off-diagonal chain; the
   `MODEL.md` chain must win decisively, else stop and re-derive. Armchair
   sign chains are never trusted over the ritual (`SYMBOLOGY.md` §7.5;
   the ASPECT `stress`-field episode).
3. **Closure is the pipeline validator, not a result.**
   $\Delta N_D - \Delta\mathrm{GPE}^* + F_B = 0$ holds for any
   equilibrium stress field; closing it validates extraction. Closure
   quality is reported wherever the balance is computed (single step and
   per-step in time series).

## 2. Reference conventions (ruled by Dan, 2026-08-31)

1. **Δ resultant curves are trench-referenced, only ever.**
   $\Delta f(x) = f(x) - \bar f(x_T)$ (or offset by $N_D(x_T)$ in the
   absolute form). No other curve reference is permitted — outboard
   anchoring of the resultants (tried in ASPECT, withdrawn) adds
   interpretive complexity for no physical gain.
2. **The register-Δ scalar** is the two-point value
   $(\cdot)(x_I) - (\cdot)(x_T)$ (POLICY_symbols), computed with declared
   windows at both ends; it is a measured quantity, independent of any
   display gauge. The incremental (curve) Δ is defined at first use per
   `SYMBOLOGY.md` §5/§7.
3. **The closure — and only the closure — is pinned.** The trench column is
   the worst-resolved in every model, so the trench-referenced closure
   carries a constant that is anchor noise, not physics. The closure is
   pinned to zero over a declared **mid-subducting-plate window** (stable,
   clear of the outer rise and of ridge/boundary influence), and the removed
   constant is **printed**. Publication framing: *the models do what they
   do; extraction and closure carry useful information but do not work
   perfectly, particularly at the trench; the displayed closure shows the
   residual's spatial structure with the trench-anchor constant declared
   and removed.*

   | repo | pinning window | status |
   |---|---|---|
   | aspect | $x_T$ + 1,500..2,500 km | implemented |
   | fluidity | $x_T$ + 1,000..2,000 km (ridge at ~3,400 km) | backport pending |
   | mdoodz | $x_T$ + 500..1,000 km (seaward extent ~1,300 km) | backport pending |

## 3. Key columns

1. **Trench $x_T$ — the fore-arc trap rule.** OP-side lows can exceed the
   trench depth (fluidity: 3,254 vs 1,717 m; aspect similar), so a pick
   that can see the landward side is unsafe *unless the margin is verified*.
   Two compliant patterns:
   - **guarded pick**: anchored on the plate-boundary velocity crossing with
     a directional/asymmetric search (fluidity `pick_trench_3step`; aspect
     `pick_trench_topo`, window $[x_v - 10, x_v + 150]$ km);
   - **verified argmin**: a global minimum is acceptable where the fore-arc
     margin is demonstrated and printed. MDOODZ status, verified
     2026-08-31: global topo argmin = trench at **every** step, margin
     350–450 m (MD7_R0) and ~2,000 m (MD7_A1). New runs must re-verify
     (one-line printed check) or adopt a guard.
2. **First isostatic column $x_I$**: first zero-crossing of $dV/dx$ seaward
   of the trench (skip ~10–20 km). Cross-checks (fs-recovery picker) welcome.

   ⚠ **CORRECTED 2026-10-10 (§10 procedure): fluidity extracts the
   fs-recovery route, not the $dV/dx$ zero.** This rule named the
   derivative crossing as the definition and the surface picker as an
   optional cross-check; `find_first_isostatic_column` — the only $x_I$
   implementation in the repo, used by the caches, `balance_chain` and the
   notebooks alike — does the opposite. It walks seaward from $x_T$ past a
   30 km buffer and returns the first column where the free surface has
   recovered to its elevation 500 km seaward. The register and the code
   have disagreed for as long as both have existed; found by external audit.

   **The rule now follows the code, because the surface route is the better
   one**: it needs no derivative of a noisy field, and $V$ has to be
   smoothed over 10 km before $dV/dx$ is usable at all. The derivative
   crossing is retained as the stated **cross-check**, and the two agree:
   2.0 km (STD) / 0.6 km (WAL) at the reference epoch measured here, and
   over all 37 retained snapshots of both runs a median 2.5 / 0.3 km, p90
   ≈ 5 km, worst 13.3 / 11.0 km, with a crossing present at every snapshot.
   Moving the 500 km far-reference to 400–700 km shifts $x_I$ by under
   11 km; 300 km moves it up to 19 km, so the reference stays at 500 km.

   *Per-repo:* fluidity as above. mdoodz and aspect declare their own
   routes in their notebooks; neither is re-opened here. **Nothing is
   recomputed** — this change makes the register describe the extraction
   that produced every committed number, and no value moves.
3. **$x_m$**: zero crossing of $V$ between $x_T$ and $x_I$ (maximum bending
   moment); fallback midpoint.
4. **Ridge**: model-specific, declared. Fluidity and aspect both carry
   **emergent** ridges (they initiate at the model corners and migrate
   inward; sidewalls free-slip, no inflow) — fluidity's on the subducting
   plate; aspect has one per plate (OP-side ridge inboard by the reference
   epoch, SP-side ridge still at/near the right corner). Ridge-push
   register-Δ uses the subducting-plate ridge column. MDOODZ (forced,
   velocity-regulated) has no ridge — a feature of the buffered end
   member, not a gap.

   **RULE CHANGE, ruled by Dan 2026-09-26, specified 2026-09-27
   (fluidity; §10 procedure).** The ridge column is the **local
   topographic maximum within ±20 km of the flow pick**, not the flow
   pick itself. Two-step, because each step fixes what the other gets
   wrong:
   - *locate* the spreading axis by the maximum surface divergence
     (`find_ridge_x_flow`). A global surface pick is unusable: measured
     across the run it sits a median 22 km (STD) / 23 km (WAL) from the
     flow pick but with a p90 of 79 / 309 km and excursions to 331 km,
     and it once flipped 313 km in a single step.
   - *refine* to the highest point of the smoothed free surface within
     ±20 km. The divergence maximum IS the spreading axis, and the
     spreading axis sits in the **axial valley**: measured on six
     snapshots per run, the local crest is 50–350 m higher and lies a
     median 5–20 km away, **systematically inboard** (negative offset on
     9 of 12 snapshots).

   **Why ±20 km.** Measured at ±10 / 20 / 30 / 50 km: the gain from 20 to
   30 km is under 40 m in every snapshot, while ±50 km escapes to a
   neighbouring high in 1 of 6 STD and 3 of 6 WAL snapshots — the failure
   mode the flow pick was adopted to avoid. ±20 km takes essentially all
   of the available rise without reintroducing it.

   **Per-repo implication.** Fluidity only. It changes the ridge column,
   so every ridge-referenced quantity moves: ridge push, the tilt
   partition and its suppression fraction, the trench-pull:ridge-push
   ratio, $N_D(x_R)$, and the isostatic-domain figures. It requires a
   rebuild of `column_profiles_cache` AND of `lab_kinematics_cache`,
   which reads `xR` from the first and averages $x_I$→ridge.
   `fig_headline_tracking` is independent (min–max normalisation).
   ASPECT is unaffected — its SP-side ridge sits at the model corner and
   is picked from the smoothed raw surface already (§6b.4). MDOODZ has no
   ridge.

   **Measured after implementation (2026-09-27).** Ridge push run medians
   1.32 → **1.52** TN/m (STD, +15 %) and 1.45 → **1.51** (WAL, +4 %);
   ΔGPE\* total 3.34 → 3.57 and 3.08 → 3.19; suppression fraction 0.33 →
   0.31 and 0.19 → 0.18. Trench pull is unchanged (2.11 / 1.74) — it is
   referenced to $x_T$ and $x_I$, neither of which moved. W32's +2 to
   +16 % was a fair bracket.

   ⚠⚠ **THE REFINEMENT RETRACTED AN APPARENT RESULT.** On the valley pick
   the ridge column appeared to carry ~22 MPa of depth-varying
   non-lithostatic support, read as dynamically supported (flexural)
   topography, and its surface seemed to supply only 60 % (STD) / 70 %
   (WAL) of the load its own density structure demands. On the crest pick
   the residual is FLAT — 49.6→47.8 MPa (STD), 49.4→48.8 (WAL), i.e.
   constant to ~3 MPa, which is exactly the elevation mass
   ($\rho g \times 1.48$ km = 47.9 MPa) — and the surface supplies
   104 % / 100 % of the density demand. **Both columns are lithostatic to
   a few MPa once elevation is accounted for.** The earlier reading was an
   artefact of sampling the axial valley.

## 4. Column values, windows, noise

1. Column quantities are **window-averaged with a uniform ±5 km half-width
   across the suite** — RULED by Dan 2026-09-01, winding back the earlier
   ±25 km standard. Justification: all three models resolve ~1 km where
   the columns live (mdoodz/aspect regular 1 km grids; fluidity AMR
   measured 2026-09-01: median triangle edge 1.1 km near the trench,
   p90 2.4 km — the coarse 46 km cells are deep-mantle only), so ±5 km
   spans 4–5 native cells everywhere. The instability that motivated
   ±25 km is OP-only, and the window-sensitivity probe showed the wide
   window *biases trench pull low* by swallowing the flexural interval —
   it was not conservative.
2. **±25 km is retired as a standard**; it may appear only as a quoted
   sensitivity alongside ±5 km values. A **window-sensitivity panel**
   (resultants vs half-width at the reference snapshot) belongs in each
   repo's further_analysis. Fluidity's time-series loop point-samples
   columns on its 200 m analysis grid (declared, §9) — finer than the
   window, not wider.
3. Trench-column values carry a **pick-sensitivity statement** where the
   axis zone is noisy — meaning the km-scale jitter of the axis pick itself
   plus window sensitivity. **The $x_v$ column is NOT a pick-sensitivity
   proxy** (corrected 2026-09-01): it is the plate-boundary trace at 8 km
   depth, ~40 km landward of the trench by interface geometry
   (offset ≈ 8 km / tan(dip), and its smooth evolution is tracked as the
   fore-arc-flip tripwire), so differences between the two columns are
   lateral structure, never pick uncertainty. The former $x_T$-vs-$x_v$
   display band conflated the two and is removed.

### 4b. Reference snapshots (PER-MODEL; mid-run rule ruled by Dan 2026-09-15)

Each **model** declares one **reference snapshot** — the epoch used for
single-snapshot figures, tables, and sensitivity studies. The rule: the
**run midpoint**, unless a compelling reason is declared ("a snapshot we
don't think has special meaning; roughly a time-averaged state"). The
declaration carries a representativeness check against the run medians.

| model | reference | t | check |
|---|---|---|---|
| fluidity STD | t = 40.0 Myr snapshot | 40.0 Myr | all headline quantities within 16–84th pctile of run; most 30–60th (2026-09-15) |
| fluidity WAL | t = 40.0 Myr snapshot | 40.0 Myr | same check; near-median throughout (supersedes the f10 = 20.0 Myr declaration — f10 catches WAL back-tilt at a local high, the known single-epoch hazard) |
| mdoodz MD7_R0 | `Output01000` (REF_INDEX 10) | 3.8 Myr | ≈ mid-run already |
| mdoodz MD7_A1 | undeclared (second-paper scope) | — | — |
| aspect | reference_model step 100 | 5.0 Myr | ≈ mid-run |

The mid-run **averaging window** for time-averaged snapshot figures is
36–44 Myr (fluidity; 5 archive snapshots), declared per figure.

⚠ **Compare archive times with a tolerance, not exactly (2026-10-09).**
The window was declared as 36–44 for both runs and delivered as 36–44
for STD and **38–44 for WAL** for as long as it had existed. WAL's
36 Myr output is stamped t = 35.9905 and STD's t = 36.0035, so a bare
`t >= 36.0` admitted five snapshots for one run and four for the other,
and every mid-run average in the repo quietly used a different window
per model. `derive()` now compares with a 0.1 Myr tolerance against the
2 Myr archive cadence. The only visible symptom had been a panel title
reading "avg 38–44 Myr, n=4"; nothing else announced it. Any future
window declared in model time needs the same treatment.
Notebooks stay freely re-runnable at any step (the user's parameter edits
are authoritative); the reference is the *quoted* epoch, not a lock.
Values previously quoted "at f10" remain valid as explicitly-labelled
epoch values.

## 5. Vertical conventions

1. **$z_c$** (column-integral depth) is declared per model with an
   **equilibration-depth justification** (the cumulative
   $\Delta\mathrm{GPE}^*(z)$ panel): mdoodz 100 km, **fluidity 80 km**,
   aspect 100 km. Unification by regridding is not required; the
   justification panel is.

   **FLUIDITY 75 → 80 km, ruled by Dan 2026-10-09 (§10 procedure).**
   W17 had already ruled that the defensible stopping point is the
   $\sigma_{zz}$ zero crossing — the depth at which the cumulative
   ridge-push integral is stationary, below which the integral
   accumulates the asthenospheric counterflow rather than any
   topographic signal. 75 km was STD's mid-run crossing. Measured over
   all 37 retained snapshots of both runs, the crossing sits at a median
   of **76.5 km (STD)** and **87.5 km (WAL)**, and it DEEPENS SECULARLY
   at ~9–10 km per 10 Myr (STD 55.5 → 89.5 km, WAL 68.5 → 113 km between
   the halves of the run; full range 42–104 and 48–126 km).

   *How 80 was chosen.* $z_c$ was swept 60–120 km at 1 km and the error
   against each snapshot's own crossing measured **in TN/m, not as a
   percentage of the peak** — the peak is as small as 0.66 TN/m early in
   STD, so a relative measure is dominated by division by a small
   number and discriminates nothing. Run-wide rms error:

   | $z_c$ | STD | WAL | joint |
   |---|---|---|---|
   | 75 | 0.073 | 0.102 | 0.175 |
   | **80** | 0.084 | 0.081 | **0.165** |
   | 85 | 0.105 | 0.070 | 0.175 |
   | 90 | 0.130 | 0.070 | 0.201 |

   80 km is the joint minimum, on the run-wide rms and on the mid-run
   window alike. The models pull opposite ways — STD's own optimum is
   74 km, WAL's is 87 — so 85 is WAL's number and costs STD enough to
   land back at 75's joint error.

   ⚠ **No fixed depth is right at every state** (Dan, 2026-10-09), and
   the convention does not claim one. The crossing moves by 60–80 km
   across an 80 Myr run; a single declared $z_c$ is the least-wrong
   compromise for a pair of runs, and the penalty for the alternatives
   considered is ~0.01 TN/m rms. Papers quoting these numbers should say
   the depth is declared and justified, not that it is correct.

   *What moves.* Nothing material: trench pull 2.108 → 2.121 (STD) and
   1.739 → 1.747 (WAL); ridge push 1.523 → 1.523 and 1.514 → 1.517;
   trench-pull:ridge-push ratio 0.722 → 0.718 and 0.871 → 0.868.
   $N_D(x_T)$ median moves most in relative terms, −0.696 → −0.787
   (STD) and −1.834 → −1.873 (WAL) — 0.09 and 0.04 TN/m, again a small
   denominator rather than a real change. Every qualitative claim is
   unchanged: tension-like fraction 38 % / 0 %, $N_D(x_R)$ positive at
   every step.

   *Per-repo implications.* **Fluidity only.** mdoodz (100 km) and
   aspect (100 km) are untouched — the rule remains per-model with a
   justification panel, not a suite-wide depth. Within fluidity: every
   column integral re-derives from the stored 250 km profiles, but
   $F_B$ does NOT — `column_profiles_cache` accumulates $\tau_{zx}$
   along the horizontal plane at $z_c$ at build time, so the cache
   must be rebuilt.
2. **Mechanical base** for windowed plate integrals: **fluidity 900 °C**
   (amended 2026-10-10 — every fluidity figure used 900 while this said
   950, and 900 is what is published; it was CALIBRATED to match the
   strength cap at mid-run, which must be stated wherever it is quoted
   rather than implied to be independent). The strength cap itself is now
   an absolute **25 MPa** on the carried fibre stress, not 10 % of the
   per-snapshot peak: a fraction cut the two models at different physical
   stress levels. Elsewhere the 950 °C isotherm,
   capped (mdoodz 80 km, fluidity 65 km, aspect 80 km).
3. **Neutral plane** $z_{np}$: the extremum of the cumulative fibre stress
   $C(z) = \int(\sigma_{xx}-\sigma_{zz})\,dz$ in a declared window
   (5–60 km), first-crossing pick computed alongside for comparison; pivot
   sensitivity $dM/dz_{\mathrm{pivot}} = -N_D$ printed, not assumed.

## 6. Physical constants and the missing ocean

1. $\Delta\rho = \rho_m - \rho_w$ uses each model's $\rho_m$ and
   $\rho_w = 0$ — **no suite model carries an ocean** (mdoodz 3250,
   fluidity 3300, aspect 3300 kg/m³).
2. Every single-step notebook carries the **water-equivalent trench depth**
   figure: $w \cdot \rho_m/(\rho_m - \rho_{w,\oplus})$ with
   $\rho_{w,\oplus} = 1030$ kg/m³ (factor ≈ 1.45), declared as a
   load-equivalent scaling, not a dynamically self-consistent correction —
   the guard against reading model trenches as systematically shallow.
   Status: aspect implemented; fluidity/mdoodz backport pending.

## 6b. The ridge-push partition and the back-tilt force (ruled by Dan, 2026-08-31)

The full driving decomposition has **three tracked components**: trench pull,
measured ridge push, and the **back-tilt force** — ridge push suppressed by
the asthenospheric return-flow pressure gradient tilting the plate down
toward the ridge. One can think of it as suppressed ridge push: it can
"move over" to ridge push as the channel evolves, so all three are tracked
through time. Everything is in the **σ_zz register only** — no lithostatic
or density-built alternative is ever quoted (the lithostatic pressure of a
non-equilibrated column is not the force-bearing quantity; the density
field's only sanctioned use is the optional mass-route cross-check of
$q_c$, aspect).

1. **Measured ridge push** is the register-Δ
   $\mathrm{GPE}^*(x_R) - \mathrm{GPE}^*(x_I)$, ridge column per §3.4,
   window-averaged (uniform ±5 km, §4.1).
2. **Deep residual** $q_c(x)$: the σ_zz difference relative to the $x_I$
   column, averaged over a declared **deep band below $z_c$** where
   equilibration has asymptoted, sign pressure-positive
   ($q_c = -[\bar\sigma_{zz}^{deep}(x) - \bar\sigma_{zz}^{deep}(x_I)]$),
   smoothed ~100 km (the tilt is a basin-scale signal). Each single-step
   back-tilt notebook carries the **asymptote justification panel**
   ($q_c$ vs band depth), the analog of the $z_c$ equilibration panel.
3. **Back-tilt force** $F_{tilt} = -q_c(x_R)\,z_c$ (density-free: the
   counterfactual elevation $\Delta e = -q_c/\rho g$ times its GPE*
   contribution $\rho g z_c \Delta e$ cancels ρ). Isostatic ridge push
   $= $ measured $+ \,F_{tilt}$; the suppression fraction is quoted.
4. **Per-repo instantiation:** deep band aspect 150–195 km, fluidity
   120–190 km (asymptote demonstrated per repo); ridge pick — fluidity
   `find_ridge_x` (SP ridge inboard), aspect SP-side pick = shallowest
   smoothed raw surface seaward of $x_I$ + 500 km with 10-km wall
   standoff (the SP ridge sits at/near the corner; SP surface is clean).
   MDOODZ: **not applicable** (no ridge — buffered end member).
5. **Time series** (§7) record per step: trench pull, measured ridge
   push, $q_c(x_R)$, $F_{tilt}$ — the three-component partition is a
   standing figure in each time-evolution analysis, and so is the
   **channel-flow test**: $F_{tilt}$ against $|v_{sp}|$ with both the
   level and the **increment** correlation quoted (increments are
   immune to shared secular trends and are the discriminating
   statistic).
6. Home: a dedicated `further_analysis` back-tilt notebook per repo
   (single-step anatomy: $q_c(x)$ profile, counterfactual topography,
   asymptote panel, partition bars), plus the time-series integration.

Implemented 2026-08-31 (aspect b893acf; fluidity 9c42e0e — caches
backfilled by time-matched file mapping; the loop carries the block for
fresh recomputes). Reference-epoch values (±5 km windows, 2026-09-01):
aspect +1.58 TN/m suppressed (50 %); fluidity STD +0.70 (39 %), WAL
+0.67 (46 %). **The single epoch
misleads on STD-vs-WAL**: through the run STD sustains roughly twice
WAL's back-tilt (run medians +0.65 vs +0.33 TN/m; WAL decays to
0.1–0.3 late-run) — the weak layer *does* reduce the suppression, per
the channel-viscosity picture; f10 catches WAL's oscillating decay at a
local high. Aspect holds a flat ~0.46 suppression fraction throughout.

## 7. Time series

1. Per-step reductions use the identical pickers/windows as the single step
   (one shared implementation per repo — helpers module or notebook
   functions; they must not fork).
2. Each step records a **closure-health metric** (rms about the pinned
   closure over a near-trench window) and, where applicable, the **pick
   guard** ($x_T - x_v$) — a pick flip must be visible in the figure, never
   silent (the aspect 3.5 Myr artifact is the cautionary case, recorded in
   that repo's notebook).
3. **Plate-interior velocities** (boxes far from the axis zone) ride along
   as the axis-zone-immune dynamics ground truth.
4. Masks (unformed trench, spin-up, ceiling-pinned picks) are counted and
   printed; time series are plotted in model time.
5. **The spin-up mask is DECLARED, not implicit (added 2026-09-25).**
   Fluidity: **t ≥ 8 Myr**, applied at cache build in
   `column_profiles_cache`, `lab_kinematics_cache`,
   `kinematics/slab_geometry_cache` and `fig_headline_tracking`'s own
   cache — so the snapshots are *dropped*, not masked downstream, and
   every figure reading those caches inherits the cut whether or not it
   restates it.

   | | |
   |---|---|
   | archive | 41 snapshots, 0–80 Myr, Δt = 2.0 Myr |
   | excluded | **4 snapshots: 0, 2, 4, 6 Myr** |
   | retained | 37 snapshots, 8.0–80.0 Myr |

   ⚠ **What this excludes is not incidental.** The free-sinking stage
   lies entirely inside it — the slab first meets the lower mantle at
   ~4–5 Myr (STD) and ~1.5 Myr (WAL), and the subducting plate reaches
   its highest velocity of the whole run there, an order of magnitude
   above the subsequent range (Cerpa et al. 2022). Every run-wide
   statistic in the suite is therefore a statement about the **mature,
   post-660-interaction regime**, and cumulative quantities accumulate
   from 8 Myr rather than from initiation.

   ⚠ This constant existed only as `T_MIN_MYR = 8.0` repeated across
   seven scripts until 2026-09-25, undeclared and unknown to Dan.
   **Fixed 2026-10-10** in the working repo: it is defined once, in
   `column_profiles_cache` beside `ZC_KM`, and the other ten scripts
   import it. By then it had spread to eight. Any
   future change to it must be made here first and the affected caches
   rebuilt — the excluded snapshots cannot be recovered from a cache
   without a rebuild.

## 8. Audit checklist (per notebook)

- [ ] step-0 pins printed (pressure, register, τ ritual) and passing
- [ ] Δ curves trench-referenced; closure pinned per §2.3 with constant printed
- [ ] picker compliant with §3.1 (guarded, or margin verified and printed)
- [ ] windows declared; cross-model values available at ±25 km
- [ ] $z_c$ + equilibration panel; base isotherm + cap declared
- [ ] water-equivalent depth figure present
- [ ] honesty box lists every filter, window, cap, and mask
- [ ] `MODEL.md` field_map covers every field the notebook reads

## 9. Deviations register (current)

| item | repos | state |
|---|---|---|
| back-tilt partition (§6b) | aspect, fluidity | implemented 2026-08-31 (back-tilt notebooks + partition time series); mdoodz N/A |
| closure pinning (§2.3) | fluidity, mdoodz | pending backport |
| water-equivalent figure (§6.2) | fluidity, mdoodz | pending backport |
| fore-arc margin print (§3.1) | mdoodz | pending (picker itself verified safe) |
| ridge = local topographic max within ±20 km of the flow pick (§3.4, 2026-09-27) | fluidity | **IMPLEMENTED 2026-09-27**: `find_ridge_x_flow` refines on `fs_top`; both caches rebuilt; 14 `fig_` scripts re-rendered. Ridge push +15 % (STD) / +4 % (WAL). ⚠ It also removed an apparent result — see §3.4 |
| uniform ±5 km window (§4.1–4.2, 2026-09-01) | all | implemented (aspect analyse_step/back_tilt, mdoodz additional_figs, fluidity trench-ref); fluidity time-series point-sampled on the 200 m grid (declared) |
| notebooks still in the mirrored frame (§1.1) | fluidity | ⚠ OPEN. Scripts and caches are native; the eight notebooks are not. They feed no paper figure. Convert them — the convention is single-frame |
| $x_I$ register/code mismatch (§3.2) | fluidity | RESOLVED 2026-10-10: the register named the $dV/dx$ zero as the definition while the only implementation uses free-surface recovery. Rule changed to follow the code; the derivative crossing is the stated cross-check and agrees to a median 2.5 / 0.3 km. No value recomputed |
| canonical ridge picker not used everywhere (§3.4) | fluidity | FIXED 2026-10-10: `balance_chain` still called the retired global topographic `find_ridge_x`, so the snapshot figures used a different ridge from the caches (median 6 / 13 km apart on the corrected surface). Now `find_ridge_x_flow`; `find_ridge_x` survives only as the `xR_topo` diagnostic |
| $z_c$ = 80 km (fluidity) | fluidity | ruled 2026-10-09, raised from 75 km: the joint minimum of the rms error against the measured per-snapshot $\sigma_{zz}$ crossing in both runs (§5.1). Not a deviation but a declared per-model value; the crossing itself moves 42→126 km across the run and no fixed depth tracks it |
| $w$ 12-km-filtered (aspect) | aspect | permanent, instability-driven, raw shown alongside |
| spin-up cut differs (§7.5) | fluidity | ⚠ the bending caches (`notebooks/further_analysis/outputs/bending_moment_*.npz`) run from **2.0 Myr** and `fig_compliance_time` / `fig_flexure_kinematics` cut at **t > 3 Myr**, inherited from the synthesis prototype — not the suite's 8 Myr. Those figures therefore include 3 snapshots every other figure excludes. Second-paper material (W9, held), so not currently paper-blocking; reconcile before any of it is quoted alongside the main set. |
| t ≤ 10.0 Myr regime mask (aspect) | aspect | ruled 2026-09-01: the slab-top coupling transition starts a distinct regime sampled ~0.3 Myr — excluded from figures/statistics, kept in cache, mask printed (further_analysis/aspect_10myr_event.ipynb) |

## 10. Change procedure

1. Edit this file at the umbrella root.
2. State the rule change and its per-repo implications here **before**
   touching any repo (Dan's consistency directive, 2026-08-31).
3. Implement repo by repo, one repo per commit, citing the convention.
4. Update the deviations register (§9) in the same change.
