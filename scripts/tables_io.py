"""Table writing and reading — the verifiable-numbers contract.

Pattern copied from `trench_pull_ferrite` (see the umbrella's
VERIFIABLE_NUMBERS_HANDOFF.md). The rule it enforces:

    a table is written by the SAME script, in the SAME pass, from the
    SAME in-memory arrays as the figure it accompanies.

Nothing is computed twice, so a table and its figure can only disagree
if one of them was not regenerated — which a harness can check.

Two functions, and they are the only way tables are written or read:

    write_table(name, fields, rows, script, figure, models, meta)
        -> tables/<name>.csv    plain CSV, header row then rows, nothing
                                else (GitHub and spreadsheets must render it)
        -> tables/<name>.json   provenance sidecar: script, figure, inputs,
                                and any scalar metadata

    read_table(name) -> (meta, rows)
        the only reader. Numbers are parsed to float. If a reader needs a
        value the table does not have, the TABLE gains a column; the
        reader never computes it.

Floats are written at six significant figures so that a rerun on another
machine produces no diff noise — otherwise every regeneration dirties the
repository and real drift becomes invisible.
"""
import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(ROOT, 'tables')
SIGFIG = 6

def _fmt(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int,)):
        return str(v)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f or f in (float('inf'), float('-inf')):
        return str(f)
    return f'{f:.{SIGFIG}g}'

def write_table(name, fields, rows, script, figure=None, models=(), meta=None):
    """Write tables/<name>.csv and its .json provenance sidecar.

    fields : sequence of column names (the header row)
    rows   : sequence of sequences, same width as fields
    script : the script that produced this table (basename)
    figure : the figure it accompanies, if any
    models : the model runs the numbers come from
    meta   : dict of scalar provenance (windows, smoothing, epochs, ...)
    """
    os.makedirs(TABLES, exist_ok=True)
    csv_path = os.path.join(TABLES, f'{name}.csv')
    with open(csv_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(list(fields))
        for r in rows:
            w.writerow([_fmt(v) for v in r])
    side = {'table': f'{name}.csv', 'script': script, 'figure': figure,
            'models': list(models), 'fields': list(fields),
            'n_rows': len(rows), 'sigfig': SIGFIG}
    if meta:
        side['meta'] = {k: (_fmt(v) if isinstance(v, float) else v)
                        for k, v in meta.items()}
    with open(os.path.join(TABLES, f'{name}.json'), 'w') as fh:
        json.dump(side, fh, indent=2)
        fh.write('\n')
    return csv_path

def read_table(name):
    """Read tables/<name>.csv -> (meta, rows).

    meta is the JSON sidecar (or {} if absent). rows are dicts keyed by
    column name, with values parsed to float where possible.
    """
    base = name[:-4] if name.endswith('.csv') else name
    meta = {}
    side = os.path.join(TABLES, f'{base}.json')
    if os.path.exists(side):
        with open(side) as fh:
            meta = json.load(fh)
    rows = []
    with open(os.path.join(TABLES, f'{base}.csv')) as fh:
        for r in csv.DictReader(fh):
            out = {}
            for k, v in r.items():
                try:
                    out[k] = float(v)
                except (TypeError, ValueError):
                    out[k] = v
            rows.append(out)
    return meta, rows

def value(name, **match):
    """Pull a single value: value('partition', model='STD', quantity='...').

    Matches on the given columns and returns the remaining single field.
    Raises if the match is not unique — a quoted number must be
    unambiguous.
    """
    _, rows = read_table(name)
    hits = [r for r in rows
            if all(str(r.get(k)) == str(v) for k, v in match.items())]
    if len(hits) != 1:
        raise KeyError(f'{name}: {match} matched {len(hits)} rows, expected 1')
    rest = [k for k in hits[0] if k not in match]
    if len(rest) != 1:
        raise KeyError(f'{name}: {match} leaves {rest}, expected one field')
    return hits[0][rest[0]]
