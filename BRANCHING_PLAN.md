# Branching plan — isolating the down-dip force-balance investigation

*Written 2026-08-20 from the git history and the notebook contents, so this survives the
loss of any single Claude conversation. Delete once executed.*

---

## 0. Housekeeping note on the `~/Downloads` references

`CLAUDE.md` points at four files in `~/Downloads/` (`slab_force_balance_roadmap.md` and
three per-cell `.py` scripts). They are no longer there. **This does not matter** — the
content lives in the notebook: the roadmap is cell 39, and the scripts' logic is in cells
27, 33 and 35–38. They were almost certainly written out by an earlier assistant session
rather than placed there deliberately.

**Action: delete the `~/Downloads` paragraph from `CLAUDE.md`** so no future session goes
looking for files that were never meant to be the source of truth. The repository is the
source of truth.

---

## 1. What §12 actually is

*(You asked. Your memory was right: it is a slab-top referenced frame.)*

**§12 — "$F_D$ along the slab-top SP boundary"** extends §8 (which evaluates $F_D$ on a
single slab-normal plane through the trench) to a *series* of slab-normal planes walking
along the slab top, from the trailing-plate lithosphere, through the trench bend, down
into the deep slab.

The pipeline: trace the subducting-plate material-fraction contour at $\mathrm{SP}=0.5$ —
the actual slab-top boundary — walk the polyline both ways from a seed near the trench,
Gaussian-smooth, fit a cubic spline, sample uniformly in arc length. At each evaluation
point take the local dip $\theta$ from the spline tangent, build a slab-normal line,
sample stresses, mask, integrate.

So yes: a **slab-aligned $(\xi,\eta)$ frame anchored on the slab-top contour**, with

$$\tau_{\xi\xi}-\tau_{\eta\eta} = (\tau_{xx}-\tau_{zz})\cos 2\theta + 2\tau_{xz}\sin 2\theta$$

Outputs are $F_D$ per plane, its horizontal ($F_D\cos\theta$) and vertical
($F_D\sin\theta$) projections, and local dip — all as functions of arc length from the
trench and of depth. Geometry comes only from the SP contour; temperature enters only in
the mask.

**It is not half-finished.** As committed it has: the pipeline, a written rationale for
the depth-decoupled masking strategy, a geometry sanity-check visualisation, the results
plots, and the full-grid rotated-stress diagnostic. Every part is documented in prose.

The "not all cells are finalised" warning in `CLAUDE.md` was written *from the Tier 3
working tree* and describes the buoyancy / interface-shear / unbent-slab / roadmap work —
all of which is Tier 3, not the committed §12.

---

## 2. The history in one table

`trench_pull_fluidity` and `egu_2026` were born three minutes apart — 2026-04-28 15:33 and
15:36 — and tracked each other for six days. The talk's last commit is
`f2b8486 "save"`, **2026-05-04 13:32**. The Fluidity commit nine minutes later is the
divergence. The poster ran two days further (`poster.typ`, 2026-05-06 04:11) but is not
under version control.

| Tier | Commits | Content | Status |
|---|---|---|---|
| **0** | `1f87d39` (05-04 13:21) | §1–§9. The state at the EGU freeze. | — |
| **1** | `93a3aeb`, `d214cfb` | §10/§11 time loop, $F_D$ vs time on three planes. | finished |
| **2** | `f0e233b` (05-05 13:44) | `cerpa_*`→`fluidity_*` rename; §12 slab-top $F_D$ (+1281); `fluidity_viscosity_evolution.ipynb` (+427). **= `main` = `origin/main`** | finished |
| **3** | *uncommitted* | Buoyancy, interface shear, unbent slab, the roadmap. +593 lines, 41 cells. | **shelve this** |

### ⚠ Tiers 2 and 3 are entangled inside §12

Tier 3 does not simply append. It **rewrites cell 27**, §12's central pipeline cell:

| cell | at HEAD (Tier 2) | in working tree (Tier 3) |
|---|---|---|
| 27 | `# ... full pipeline (sanity viz + F_D plots)` | `# ... F_D + buoyancy + interface-shear pipeline` |
| 33 | F_D plots | rewritten |
| 37 | plot cell | rewritten |
| 38 | — | **new** — unbent-slab visualisation |
| 39 | — | **new** — the roadmap |

Consequences: you cannot shelve Tier 3 by deleting trailing cells, and a future
branch→main merge *will* conflict in the middle of §12. Treat the branch as **a reference
to read from, not a branch to merge back.**

---

## 3. Decisions

**DECIDED 2026-08-20: §12 stays on `main`.** Only Tier 3 is shelved. The cleave between
"finished diagnostic" and "open investigation" falls exactly at the Tier 2 / Tier 3
boundary, which is also the only boundary git gives you for free. `main` keeps §1–§12 and
receives the symbology sweep. This supersedes the note in `CLAUDE.md` saying `main` should
be restored to §1–§9 — that note was written from the Tier 3 working tree and was
describing the investigation, not §12 as committed. **Update `CLAUDE.md` accordingly when
you make the Tier 3 commit.**

**On ordering.** Your instinct — fix symbology before splitting — is right, and it does
*not* conflict with saving Tier 3 today, because committing Tier 3 to a new branch leaves
`main` at `f0e233b`, byte for byte. It is not a content split; it is a save. Sequence:

1. **Today** — commit Tier 3 to `downdip-force-balance`, push it. `main` untouched.
2. **Then** — reset the working tree, do the symbology decisions and sweep on `main`
   with nothing else in flight.
3. **Then** — tag and deposit `main`.
4. **Whenever** — revive the branch; run `/symbology-audit` on it at that point.

**Zenodo is safe either way.** Zenodo's GitHub integration archives a *release*, which
packages the tree at one tag. Side branches are never included. Pushing
`downdip-force-balance` costs you nothing in the deposit.

---

## 4. Commands — run these in your own terminal

Not through the Claude desktop bridge: git *writes* over that mount leave `.lock` and
`tmp_obj_*` debris behind, because the mount forbids unlink.

```bash
cd ~/projects/pyvista/trench_pull/trench_pull_fluidity

# 1. Save Tier 3. main stays at f0e233b.
git switch -c downdip-force-balance
git add -A notebooks/ CLAUDE.md BRANCHING_PLAN.md
git commit -m "WIP: down-dip force balance — slab-top F_D, buoyancy, interface shear

Rewrites the section 12 pipeline to add cumulative down-dip buoyancy (W_cum) and
an analytical interface-shear term (F_int); adds the unbent-slab visualisation and
cell 39, a roadmap for the complete slab-parallel balance.

Predates the SYMBOLOGY.md notation standard - run /symbology-audit before merging.
Do not merge blind: cell 27 diverges from main inside section 12."
git push -u origin downdip-force-balance

# 2. Return main to a clean f0e233b.
git switch main
git status          # expect: clean, plus untracked .claude/
```

Then add `.claude/settings.local.json` to `.gitignore` and commit the rest of `.claude/`
if you put project skills there.

---

## 5. Still outstanding elsewhere

- `egu_2026` — 12 uncommitted files including `talk.qmd`, `talk_slides.pdf`,
  `talk_notes.pdf`. The talk as delivered was never committed.
- `egu_2026_poster` — no git at all, holds two days of work past the talk. Needs
  `git init` before the symbology sweep touches it.
- `trench_gpe_paper` — 14 uncommitted files.
- `egu_2026_poster/figures/figures` — dangling symlink; should be
  `../../egu_2026/figures`. Pre-existing, unrelated to the move.
- `~/projects/pyvista/trench_pull_git_debris/` — lock files from the umbrella `git init`,
  safe to delete.
- `~/projects/pyvista/_to_delete/` — staging copies, safe to delete.
