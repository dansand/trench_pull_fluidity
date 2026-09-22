"""fig_compliance_time — does the elastic w_T(V, M) model explain the VARIABILITY?

Writes figures/kinematics/fig_compliance_time.png and
tables/compliance_time.csv from the committed bending caches
(notebooks/further_analysis/outputs/bending_moment_{STD,WAL}.npz).

Requested by Dan 2026-09-22. The compliance prototype in the umbrella's
synthesis/ fitted this model across all five runs on 2026-08-31:

    w_T = [ C_V |V_T| alpha + C_M |M_T| ] / (Delta rho g alpha^2)

with alpha solved per step so the elastic first zero-crossing matches the
measured x_I - x_T. An elastic broken plate gives (C_V, C_M) = (2, 2).
Recorded fits: MDOODZ (2.23, 1.83) R2 = 0.94; ASPECT (2.99, 2.89)
R2 = 0.81; Fluidity STD+WAL jointly (0.50, 2.13) R2 = 0.75 -- the C_V
collapse that survives every extraction variant.

This figure does two things that the prototype did not. It fits STD and
WAL SEPARATELY (Dan, 2026-09-22), and it asks the question that matters
after a day of finding trend-driven correlations:

  ⚠ DOES THE MODEL EXPLAIN THE VARIABILITY, OR ONLY THE LEVEL?

w_T, V and M all trend together over 80 Myr, so a high R2 on the raw
series can coexist with a model that explains none of the fluctuation.
Row 3 therefore repeats the comparison on LINEARLY DETRENDED series and
reports a separate R2. The gap between the two is the result.

  1  the inputs: V_T and M at the trench through time.
  2  measured w_T against the fitted model, absolute, with R2, plus the
     two model terms (C_V and C_M) separately so their relative work is
     visible and any change through the run shows.
  3  the same comparison detrended, with its own R2.

⚠ The neutral-plane key in the bending caches is h_np, not z_np. The
synthesis prototype (synthesis/compliance_fit_grid.py) still reads
f["z_np"] and would now raise KeyError on the current caches -- the
SYMBOLOGY 2026-09-01 rename z_np -> h_np broke it and nobody re-ran it.
Fix there before reusing that script.

Related: PAPER_PLAN W9 holds the w_T(V, M) elastic-fit thread for the
SECOND paper, so this figure is groundwork rather than v1 material.

DRAFT CAPTION. The elastic compliance model for trench deflection through
both runs, STD (left) and WAL (right), fitted separately. Top: the
vertical shear resultant and bending moment at the trench. Middle: the
measured trench deflection against the model, with the shear and moment
contributions shown separately. Bottom: the same comparison with linear
trends removed from both series, which tests whether the model captures
the fluctuations rather than only the secular level.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from tables_io import write_table

ROOT = os.path.dirname(os.path.dirname(_HERE))
DRHO, G = 3300.0, 9.8          # no ocean (conventions §6.1)
T_MIN = 3.0                    # spin-up cut, as in the synthesis prototype
C_MEAS = 'k'
C_MODEL = '#0072B2'
C_V_TERM = '#7B3294'
C_M_TERM = '#D55E00'


def elastic_x0(V_abs, M_abs, alpha, drho, g):
    """First zero-crossing of the elastic broken-plate deflection."""
    K = 2.0 / (drho * g * alpha ** 2)
    A = K * (V_abs * alpha + M_abs); B = -K * M_abs
    t0 = np.arctan2(A, -B) if (A or B) else np.nan
    return t0 * alpha


def prep(ts):
    good = (ts['w_T'] > 200.0) & (ts['z_np'] < 59e3) & (ts['t_myr'] > T_MIN)
    al = np.full(len(ts['t_myr']), np.nan)
    for k in range(len(al)):
        if not good[k]:
            continue
        try:
            al[k] = brentq(lambda a: elastic_x0(abs(ts['V_T'][k]), abs(ts['M'][k]),
                                                a, DRHO, G) - ts['xI_rel'][k],
                           5e3, 500e3)
        except Exception:
            pass
    ts['good'] = good & np.isfinite(al)
    ts['alpha'] = al
    return ts


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def main():
    fig, axes = plt.subplots(3, 2, figsize=(12.6, 10.2), sharex='col')
    rows = [('model', 'quantity', 'value')]

    for col, key in enumerate(('STD', 'WAL')):
        f = dict(np.load(os.path.join(
            ROOT, 'notebooks', 'further_analysis', 'outputs',
            f'bending_moment_{key}.npz')))
        ts = prep(dict(t_myr=f['t_myr'], w_T=f['w_T'], V_T=f['V_T'], M=f['M'],
                       z_np=f['h_np'], xI_rel=f['x_I'] - f['x_T']))
        g = ts['good']
        t, a = ts['t_myr'][g], ts['alpha'][g]
        V, M, w = np.abs(ts['V_T'][g]), np.abs(ts['M'][g]), ts['w_T'][g]

        # fit PER MODEL, in the prototype's linearised form
        A = np.vstack([V * a, M]).T
        Y = w * DRHO * G * a ** 2
        (cV, cM), *_ = np.linalg.lstsq(A, Y, rcond=None)
        w_model = (cV * V * a + cM * M) / (DRHO * G * a ** 2)
        w_V = cV * V * a / (DRHO * G * a ** 2)     # shear contribution
        w_M = cM * M / (DRHO * G * a ** 2)         # moment contribution
        R2_abs = r2(w, w_model)
        # The prototype's headline R2 is computed in the LINEARISED space,
        # Y = w drho g alpha^2, not on the deflection. alpha varies by a
        # large factor between steps, so alpha^2 dominates Y's variance and
        # that R2 mostly measures how well alpha^2 is reproduced -- not how
        # well w_T is predicted. Both are reported; they differ enormously.
        R2_lin = r2(Y, A @ np.array([cV, cM]))

        ld = lambda y: y - np.polyval(np.polyfit(t, y, 1), t)
        R2_det = r2(ld(w), ld(w_model))
        rr = float(np.corrcoef(ld(w), ld(w_model))[0, 1])

        a0, a1, a2 = (axes[r, col] for r in range(3))
        a0.plot(t, V / 1e12, '-', color=C_V_TERM, lw=1.8, label='$|V_T|$ [TN/m]')
        a0.set_ylabel('$|V_T|$ [TN/m]', fontsize=9, color=C_V_TERM)
        a0.tick_params(axis='y', labelcolor=C_V_TERM)
        a0b = a0.twinx()
        a0b.plot(t, M / 1e17, '-', color=C_M_TERM, lw=1.8)
        a0b.set_ylabel('$|M_T|$ [$10^{17}$ N]', fontsize=9, color=C_M_TERM)
        a0b.tick_params(axis='y', labelcolor=C_M_TERM)
        a0.set_title(f'{key}   fitted $C_V$ = {cV:.2f}, $C_M$ = {cM:.2f}'
                     '   (elastic: 2, 2)', fontsize=10.5)

        a1.plot(t, w, '-', color=C_MEAS, lw=2.2, label='measured $w_T$')
        a1.plot(t, w_model, '--', color=C_MODEL, lw=2.0,
                label=f'model  ($R^2$ = {R2_abs:+.2f} on $w_T$; '
                      f'{R2_lin:+.2f} linearised)')
        a1.plot(t, w_V, '-', color=C_V_TERM, lw=1.2, alpha=0.8,
                label='shear term $C_V|V|\\alpha$')
        a1.plot(t, w_M, '-', color=C_M_TERM, lw=1.2, alpha=0.8,
                label='moment term $C_M|M|$')
        a1.set_ylabel('$w_T$ [m]', fontsize=10)
        a1.legend(frameon=False, fontsize=8, loc='upper left', ncol=2)

        a2.plot(t, ld(w), '-', color=C_MEAS, lw=2.2, label='measured, detrended')
        a2.plot(t, ld(w_model), '--', color=C_MODEL, lw=2.0,
                label=f'model, detrended  ($R^2$ = {R2_det:+.2f}, $r$ = {rr:+.2f})')
        a2.axhline(0, color='k', lw=0.8)
        a2.set_ylabel('$w_T$ anomaly [m]', fontsize=10)
        a2.set_xlabel('Model time [Myr]', fontsize=11)
        a2.legend(frameon=False, fontsize=8, loc='upper left')
        for ax in (a0, a1, a2):
            ax.grid(alpha=0.25, color='#BFC3D1', lw=0.6)

        share = 100 * np.mean(w_V) / np.mean(w_V + w_M)
        print(f'{key}: C_V {cV:.2f}, C_M {cM:.2f}; R2 absolute {R2_abs:+.3f}, '
              f'R2 DETRENDED {R2_det:+.3f} (r {rr:+.2f}); shear term carries '
              f'{share:.0f} % of the mean deflection; n={g.sum()}; '
              f'R2 in the prototype linearised space {R2_lin:+.3f}')
        rows += [(key, 'C_V', f'{cV:.3f}'), (key, 'C_M', f'{cM:.3f}'),
                 (key, 'C_M_over_C_V', f'{cM/cV:.3f}'),
                 (key, 'r2_absolute', f'{R2_abs:.3f}'),
                 (key, 'r2_linearised_prototype_space', f'{R2_lin:.3f}'),
                 (key, 'r2_detrended', f'{R2_det:.3f}'),
                 (key, 'corr_detrended', f'{rr:.3f}'),
                 (key, 'shear_share_of_mean_deflection_percent', f'{share:.1f}'),
                 (key, 'n_snapshots', str(int(g.sum())))]

    fig.suptitle('Elastic compliance model for the trench deflection, fitted per '
                 'model — absolute fit versus detrended fit', fontsize=11.5)
    fig.tight_layout()
    out = os.path.join(ROOT, 'figures', 'kinematics', 'fig_compliance_time.png')
    fig.savefig(out, bbox_inches='tight', dpi=220)
    print('written:', out)
    tab = write_table('compliance_time', rows[0], rows[1:],
                      script='kinematics/fig_compliance_time.py',
                      figure='kinematics/fig_compliance_time.png',
                      models=('STD', 'WAL'),
                      meta={'model': 'w_T = [C_V |V| alpha + C_M |M|] / (drho g alpha^2)',
                            'elastic_reference': '(C_V, C_M) = (2, 2)',
                            'alpha': 'per step, from the measured x_I - x_T',
                            'fit': 'per model, least squares, linearised form'})
    print('written:', tab)


if __name__ == '__main__':
    main()
