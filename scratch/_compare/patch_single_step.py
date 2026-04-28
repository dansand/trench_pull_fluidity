"""Patch the pick_trench_3step cell in cerpa_single_step.ipynb in place.

Targets the single cell whose source defines `pick_trench_3step`. Replaces
the leftmost-rule with a subducting_side-aware branch (leftmost for 'left',
rightmost for 'right'), keeping all other content unchanged.
"""
import json
from pathlib import Path

NB = Path("/Users/DSAND/projects/pyvista/trench_pull_fluidity/notebooks/cerpa_single_step.ipynb")

OLD = """        # 🔑 NEW RULE: pick leftmost velocity transition
        x_v = float(np.min(xc))
        vx_method = "zero_cross_leftmost"
"""

NEW = """        # Pick the velocity zero-crossing on the subducting-plate side.
        # Leftmost when the slab descends to the left (subducting_side='left'),
        # rightmost when it descends to the right (subducting_side='right').
        # Symmetric with step 3's directional pressure search; without this
        # branching, mirrored runs lock onto the wrong crossing in snapshots
        # with multiple zero-crossings and the trench jumps landward.
        if subducting_side == "right":
            x_v = float(np.max(xc))
            vx_method = "zero_cross_rightmost"
        else:
            x_v = float(np.min(xc))
            vx_method = "zero_cross_leftmost"
"""

nb = json.loads(NB.read_text())
patched = 0
for c in nb["cells"]:
    if c.get("cell_type") != "code":
        continue
    src = "".join(c.get("source", []))
    if "def pick_trench_3step" not in src:
        continue
    if OLD not in src:
        raise SystemExit(f"old pattern not found in cell {c.get('id')!r}")
    new_src = src.replace(OLD, NEW)
    # store as list of lines preserving newlines, like nbformat does
    lines = new_src.splitlines(keepends=True)
    c["source"] = lines
    patched += 1

if patched != 1:
    raise SystemExit(f"expected exactly 1 picker cell, patched {patched}")

NB.write_text(json.dumps(nb, indent=1))
print(f"patched {patched} cell in {NB.name}")
