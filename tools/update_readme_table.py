"""Regenerate the README's figure-provenance table from the manuscript.

    python tools/update_readme_table.py [--check]

WHY THIS EXISTS HERE. The README's central claim is that every figure in
the paper traces to a named script and a CSV of the numbers in its
caption. The table carrying that claim is DERIVED from the manuscript's
own \\includegraphics calls, so it cannot be wrong about what the paper
contains -- but only if something regenerates it.

It was generated once, by the migration script that built this repo, and
that script is retired. Three SI figures were added afterwards and the
table silently fell behind, which is precisely the failure the table was
meant to be immune to. Run this after adding or removing a paper figure;
--check exits non-zero if the table is stale, which is the cheap way to
notice.

The manuscript is read-only from here: this only ever writes README.md.
"""
import argparse, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
MS = '/Users/DSAND/projects/gpe_geodynamica/2026_manuscript'


def table():
    rows, n = [], 0
    main = open(f'{MS}/main.tex').read()
    for sec in re.findall(r'^\\input\{sections/([a-z_]+)\}', main, re.M):
        for g in re.findall(r'includegraphics[^{]*\{figures/([A-Za-z0-9_.]+)',
                            open(f'{MS}/sections/{sec}.tex').read()):
            n += 1
            rows.append((f'Fig {n}', g))
    for n, g in enumerate(re.findall(
            r'includegraphics[^{]*\{figures/([A-Za-z0-9_.]+)',
            open(f'{MS}/si.tex').read()), 1):
        rows.append((f'Fig S{n}', g))
    out = []
    for lab, g in rows:
        p = os.path.join(REPO, 'scripts', f'{g}.py')
        if not os.path.exists(p):
            out.append(f'| {lab} | `{g}` | — | — |')
            continue
        t = re.findall(r"write_table\(\s*'([a-z_0-9]+)'", open(p).read())
        out.append(f'| {lab} | `figures/{g}.png` | `scripts/{g}.py` | '
                   + (f'`tables/{t[0]}.csv` |' if t else '— |'))
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    p = os.path.join(REPO, 'README.md')
    s = open(p).read()
    new = table()
    i = s.index('| Paper | Figure file | Script | Numbers |')
    j = s.index('\n\n', i)
    head = s[i:s.index('\n', s.index('\n', i) + 1) + 1]   # header + separator
    cur = s[len(head) + i:j]
    if cur.strip() == new.strip():
        print(f'README table up to date ({len(new.splitlines())} figures)')
        return
    if a.check:
        print(f'STALE: README lists {len(cur.strip().splitlines())} figures, '
              f'the manuscript has {len(new.splitlines())}')
        sys.exit(1)
    open(p, 'w').write(s[:len(head) + i] + new + s[j:])
    print(f'README table regenerated: {len(new.splitlines())} figures')


if __name__ == '__main__':
    main()
