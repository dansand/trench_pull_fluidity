"""fig_budget_time — the three-term force budget through time.

Writes figures/fig_budget_time.png and tables/budget_time.csv from the
committed column-profile cache. The temporal complement to
fig_balance_snapshot: that figure shows the three terms against distance
at one epoch, this one shows them against time for the whole trailing
plate (trench to ridge).

Requested by Dan 2026-09-20; the manuscript architecture records the same
gap ("None of the current figures shows all three balance terms through
time"). fig_partition_time is not a substitute — it divides the GPE-like
force into physical components rather than plotting the balance terms.

SIGN CONVENTION (Dan's, 2026-09-20). Every curve is that term's
contribution to the net force in +x, the analysis frame's rightward
direction. Because the subducting plate lies to the right and moves
trench-ward, that means:

    negative = force to the left  = DRIVING
    positive = force to the right = RESISTING

The balance is dN_D - dGPE* + F_B = 0 between the trench and ridge
columns, so the three plotted curves are dN_D, -dGPE* and F_B, and they
sum to the closure. Run medians: the GPE-like term drives at -3.34 (STD)
/ -3.08 (WAL) TN/m; dN_D resists at +0.94 / +2.02 and basal traction at
+2.58 / +1.22. Closure rms is 1.4 % / 1.1 % of the GPE-like term.

NOTE the sign convention differs from fig_ridge_column_stresses panel
(c), where positive means driving. The two should be reconciled before
both appear in one manuscript.

Colours follow the balance palette (FIGURE_STYLE.md): N_D black,
GPE-like term blue, F_B red, closure green dashed.

DRAFT CAPTION. The three terms of the vertically integrated force
balance between the trench and ridge columns, through the runs, for STD
(left) and WAL (right). Each curve is that term's contribution to the
net horizontal force, signed so that negative acts toward the trench and
drives the plate while positive resists. The GPE-like force is the only
driving term; the normal-stress-difference resultant and the accumulated
basal traction both resist. Their sum (green dashed) is the closure.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import column_profiles_cache as cpc
from tables_io import write_table

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C_RULE = '#BFC3D1'
T_MIN_MYR = 8.0

def terms(d, key):
    """The three balance terms between the trench and ridge columns,
    each as its contribution to the net force in +x."""
    if not hasattr(np, 'trapz'):
        np.trapz = np.trapezoid
    z = d['z']
    zc = z <= cpc.ZC_KM * 1e3
    r = cpc.resultants(d, key)
    szz_T = np.trapz(d[f'{key}_szz_T'][:, zc], z[zc], axis=1)
    szz_R = np.trapz(d[f'{key}_szz_R'][:, zc], z[zc], axis=1)
    d_gpe = -(szz_R - szz_T)                 # the GPE-like force
    d_nd = r['nd_R'] - r['nd_T']
    f_b = d[f'{key}_FB_R']
    return dict(t=r['t'], d_nd=d_nd, gpe_term=-d_gpe, f_b=f_b,
                closure=d_nd - d_gpe + f_b, d_gpe=d_gpe)

def main():
    d = cpc.load()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), sharey=True, sharex=True)
    rows = [('model', 'quantity', 'value')]
    for ax, key in zip(axes, ('STD', 'WAL')):
        q = terms(d, key)
        m = q['t'] >= T_MIN_MYR
        t = q['t'][m]
        ax.axhline(0, color='k', lw=1.6)
        ax.plot(t, q['gpe_term'][m] * 1e-12, color='b', lw=4, alpha=0.6,
                label=r'$-\Delta\mathrm{GPE}^{*}$')
        ax.plot(t, q['d_nd'][m] * 1e-12, color='k', lw=1.6, label=r'$\Delta N_D$')
        ax.plot(t, q['f_b'][m] * 1e-12, color='red', lw=2, label=r'$F_B$')
        ax.plot(t, q['closure'][m] * 1e-12, color='g', ls='--', lw=2.5,
                label='closure')
        ax.set_title(key, fontsize=11)
        ax.set_xlabel('Model time [Myr]', fontsize=11)
        ax.grid(alpha=0.25, color=C_RULE, lw=0.6)
        med = lambda a: np.median(a[m]) / 1e12
        print(f'{key}: -dGPE* {med(q["gpe_term"]):+.2f}, dN_D {med(q["d_nd"]):+.2f}, '
              f'F_B {med(q["f_b"]):+.2f}, closure {med(q["closure"]):+.3f} TN/m')
        for name, arr in (('gpe_like_force', q['d_gpe']), ('delta_nd', q['d_nd']),
                          ('basal_traction', q['f_b']), ('closure', q['closure'])):
            a = arr[m] / 1e12
            rows += [(key, f'{name}_median_TNm', f'{np.median(a):.3f}'),
                     (key, f'{name}_q1_TNm', f'{np.percentile(a, 25):.3f}'),
                     (key, f'{name}_q3_TNm', f'{np.percentile(a, 75):.3f}')]
        rows += [(key, 'closure_rms_TNm', f'{np.sqrt(np.mean((q["closure"][m]/1e12)**2)):.3f}'),
                 (key, 'closure_rms_percent_of_gpe',
                  f'{100*np.sqrt(np.mean((q["closure"][m])**2))/np.median(np.abs(q["d_gpe"][m])):.2f}')]
    axes[0].set_ylabel('Force per unit distance [TN/m]', fontsize=11)
    axes[0].legend(frameon=False, fontsize=9.5, loc='center left', ncol=2)
    for ax, lab, va, dy in ((axes[1], 'resisting', 'top', 0.97),
                            (axes[1], 'driving', 'bottom', 0.03)):
        ax.text(0.985, dy, lab, transform=ax.transAxes, ha='right', va=va,
                fontsize=9, style='italic', color='0.4')
    fig.suptitle('The three balance terms between the trench and ridge columns, '
                 'through time', fontsize=11)
    fig.tight_layout()
    out = os.path.join(ROOT, 'figures', 'fig_budget_time.png')
    fig.savefig(out, bbox_inches='tight', dpi=220)
    print('written:', out)
    tab = write_table('budget_time', rows[0], rows[1:],
                      script='fig_budget_time.py', figure='fig_budget_time.png',
                      models=('STD', 'WAL'),
                      meta={'domain': 'trench to ridge columns',
                            'zc_km': cpc.ZC_KM, 't_min_Myr': T_MIN_MYR,
                            'sign': 'positive = +x = resisting; negative = driving'})
    print('written:', tab)

if __name__ == '__main__':
    main()
