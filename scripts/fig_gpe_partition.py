"""fig_gpe_partition — the driving term decomposed: trench pull, ridge push, tilt.

Writes figures/fig_gpe_partition.png and tables/gpe_partition.csv from the
committed column-profile cache.

Requested by Dan 2026-09-24 after an audit found no existing figure did
this. The paper's second half is a deeper dive into the GPE-like driving
force, and the decomposition it needs is two levels:

    Delta GPE*  =  trench pull  +  ridge push          (by domain)
    ridge push  =  isostatic (density) part  -  tilt   (within the ridge domain)

`fig_gpe_decomposition` carried only the first level; `fig_partition_time`
(SI) carried only the second, with trench pull demoted to a grey "for
scale" line and no total. This figure carries both, and is intended to
replace `fig_partition_time` in the SI rather than sit beside it.

  1  the decomposition. Black: the plate-wide driving term. Blue: the
     non-isostatic (trench pull) domain. Orange solid: the ridge push as
     it enters the balance. Orange dashed: the same domain with the tilt
     removed -- the isostatic part the density structure supplies over
     the same depth. The purple band between them IS the tilt.
  2  the suppression fraction, tilt / isostatic part -- the channel dial.

⚠ ONE INTEGRATION DEPTH THROUGHOUT (z_c), SO EVERY LEVEL IS EXACTLY
ADDITIVE. The reader will check that blue + orange sums to black, and it
does, to machine precision -- asserted before rendering. The tilt is
evaluated on the analysis grid rather than at the nominal z_c (the cache
z is cell-centred, 500 m .. 74,500 m, so the column integral spans 74 km
and the nominal 75 would break additivity by 1.3 %).

⚠ This differs deliberately from `cpc.partition`, which integrates the
ridge side to z*, the sign-change depth, and reports a STATIC ridge push
read off the deep plateau. Both are sanctioned -- conventions W17 records
that the two agree within ~2 % -- but partition's quantities are not
mutually additive with a z_c-based trench pull, and additivity is the
whole point here. The static limit is drawn as a faint dotted line for
continuity with the SI figure; the gap between it and the dashed curve is
the density tail below z_c, which does not enter the balance.

⚠ NO RESIDUAL PANEL, deliberately. Trench pull and ridge push are both
referenced to the first isostatic column, which cancels from their sum
and therefore enters them with opposite sign; that column is the noisiest
in the cache, so their INDIVIDUAL short-timescale variability is
contaminated (DYNAMICS_FINDINGS §5.10). Levels and secular trends are
unaffected -- the contamination is high-frequency -- so this figure is
clear of the problem, and must stay that way. Do not add a detrended row.

WHAT IT SHOWS.

  - **The trench pull is the larger domain.** Medians 2.11 against 1.32
    (STD) and 1.74 against 1.45 (WAL) TN/m, out of totals of 3.34 and
    3.08. In STD the ridge domain never overtakes it (0 % of snapshots);
    in WAL it does on 22 %, late in the run.
  - The secular rise is nevertheless mostly isostatic -- ridge push
    supplies 64 % (STD) / 72 % (WAL) of it (fig_gpe_decomposition, same
    cache). The two domains divide the labour: trench pull holds the
    larger LEVEL, ridge push supplies the larger GROWTH.
  - **The tilt removes a large and steadily DECLINING fraction of the
    isostatic ridge push** -- median 0.33 (STD) / 0.19 (WAL), falling
    from ~0.6-0.7 early to ~0.18 (STD) / ~0.08 (WAL) late as the channel
    evolves. The STD/WAL separation is the weak-layer dial working in the
    direction the mechanism predicts.

⚠ DENOMINATOR. The suppression fraction plotted here is tilt / isostatic
part, both at z_c, which is the conventions §6b definition and is exactly
additive with the curves above. `fig_partition_time` (SI) divides by the
STATIC ridge push instead, which is larger, so its fraction is lower.
Both are tabulated (`tilt_over_static_median`) so the two can be
reconciled; do not compare the two numbers without saying which is which.

⚠ WAL ridge-referenced quantities after 70 Myr carry the standing
ridge-pick caveat; and WAL's 62-70 Myr excursion is the loading-and-
release episode (§2c), not a cycle.

DRAFT CAPTION. The plate-wide driving term decomposed, for STD (left) and
WAL (right). (a, b) The GPE-like force between the trench and ridge
columns (black) separated into its two domains: the non-isostatic or
trench pull domain (blue) and the isostatic or ridge push domain (orange
solid), which sum to it exactly. The orange dashed curve is the ridge
domain with the plate tilt removed -- the force the cooling density
structure supplies over the same depth -- so the purple band between the
two orange curves is the tilt contained within the measured ridge push.
The faint dotted curve is the static limit obtained by integrating the
density structure below the plate, shown for reference; it does not enter
the balance. (c, d) The tilt as a fraction of the isostatic part: the
suppression of ridge push by the adverse asthenospheric pressure
gradient, and the quantity the weak asthenospheric layer changes.
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
C_TOTAL = 'k'
C_TRENCH = '#0072B2'        # non-isostatic / trench pull domain (schematic)
C_RIDGE = '#D55E00'         # isostatic / ridge push domain (schematic)
C_TILT = '#7B3294'          # tilting, as everywhere else in the suite


def main():
    if not hasattr(np, 'trapezoid'):
        np.trapezoid = np.trapz
    d = cpc.load()
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.6), sharex='col',
                             gridspec_kw={'height_ratios': [1.75, 1.0]})
    rows = [('model', 'quantity', 'value')]

    for col, key in enumerate(('STD', 'WAL')):
        c = cpc.derive(d, key)
        z, zc, t = c['z'], c['zc'], c['t']
        I = lambda a: np.trapezoid(a[:, zc], z[zc], axis=1) / 1e12
        TP = -I(c['p_T'])                       # non-isostatic domain
        RP = I(c['p_R'])                        # isostatic domain, as measured
        TOT = TP + RP                           # = Delta GPE* over 0..z_c
        TILT = -c['dP'] * (z[zc][-1] - z[zc][0]) / 1e12
        DENS = RP + TILT                        # isostatic part, same depth
        STATIC = cpc.partition(d, key)['static'] / 1e12
        supp = TILT / DENS

        assert np.allclose(I(d[f'{key}_szz_T']) - I(d[f'{key}_szz_R']), TOT,
                           atol=1e-9), f'{key}: TP + RP is not the GPE-like term'
        assert np.allclose(DENS - TILT, RP, atol=1e-9), \
            f'{key}: density part - tilt is not the measured ridge push'

        a0, a1 = axes[0, col], axes[1, col]
        a0.fill_between(t, RP, DENS, color=C_TILT, alpha=0.30, lw=0,
                        label='tilt (removed from the ridge domain)')
        a0.plot(t, STATIC, ':', color=C_RIDGE, lw=1.2, alpha=0.75,
                label='static limit (density below $z_c$; not in the balance)')
        a0.plot(t, DENS, '--', color=C_RIDGE, lw=1.6,
                label='isostatic part, tilt removed')
        a0.plot(t, RP, '-', color=C_RIDGE, lw=2.2,
                label='ridge push (isostatic domain)')
        a0.plot(t, TP, '-', color=C_TRENCH, lw=2.2,
                label='trench pull (non-isostatic domain)')
        a0.plot(t, TOT, '-', color=C_TOTAL, lw=2.6,
                label=r'$\Delta\mathrm{GPE}^{*}$ (= blue + orange)')
        a0.set_title(key, fontsize=11.5)
        a0.grid(alpha=0.25, color=C_RULE, lw=0.6)

        a1.plot(t, supp, '-', color=C_TILT, lw=2.2)
        a1.axhline(np.median(supp), color=C_TILT, ls=':', lw=1.2)
        a1.text(t[-1], np.median(supp), f'  median {np.median(supp):.2f}',
                color=C_TILT, fontsize=8.5, va='center')
        a1.set_xlabel('Model time [Myr]', fontsize=11)
        a1.grid(alpha=0.25, color=C_RULE, lw=0.6)

        # A first-crossing is ill-conditioned when the two curves run
        # close together, so report the FRACTION of snapshots on which the
        # ridge domain exceeds the trench domain instead.
        frac_ridge = float((RP > TP).mean())

        print(f'{key}: medians  total {np.median(TOT):.2f}  trench pull '
              f'{np.median(TP):.2f}  ridge push {np.median(RP):.2f}  '
              f'isostatic part {np.median(DENS):.2f}  tilt {np.median(TILT):.2f}'
              f'  static {np.median(STATIC):.2f} TN/m;  suppression median '
              f'{np.median(supp):.2f} (range {supp.min():.2f}-{supp.max():.2f});'
              f'  tilt/static median {np.median(TILT / STATIC):.2f};'
              f'  ridge push exceeds trench pull at {100*frac_ridge:.0f} % of steps')
        for nm, y in (('total_gpe_like', TOT), ('trench_pull', TP),
                      ('ridge_push_measured', RP), ('isostatic_part', DENS),
                      ('tilt', TILT), ('static_limit', STATIC),
                      ('suppression_fraction', supp)):
            rows += [(key, f'{nm}_median', f'{np.median(y):.4f}'),
                     (key, f'{nm}_q1', f'{np.percentile(y, 25):.4f}'),
                     (key, f'{nm}_q3', f'{np.percentile(y, 75):.4f}')]
        rows += [(key, 'fraction_steps_ridge_exceeds_trench', f'{frac_ridge:.3f}'),
                 (key, 'trench_share_of_total_median',
                  f'{np.median(TP / TOT):.4f}'),
                 # tilt/static is the denominator used by fig_partition_time
                 # (SI) and PAPER_PLAN; tilt/isostatic-part is plotted here.
                 # Both are tabulated so the two figures can be reconciled.
                 (key, 'tilt_over_static_median', f'{np.median(TILT / STATIC):.4f}')]

    for row in range(2):
        lo = min(ax.get_ylim()[0] for ax in axes[row, :])
        hi = max(ax.get_ylim()[1] for ax in axes[row, :])
        for ax in axes[row, :]:
            ax.set_ylim(lo, hi)
    axes[0, 0].set_ylabel('Force per unit distance [TN/m]', fontsize=11)
    axes[1, 0].set_ylabel('tilt / isostatic part', fontsize=10)
    h, l = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(h[::-1], l[::-1], frameon=False, fontsize=8.5,
                      loc='upper left', ncol=2)
    fig.suptitle('The driving term by domain, and the tilt within the ridge domain',
                 fontsize=10.5, color='0.35')
    fig.tight_layout()
    out = os.path.join(ROOT, 'figures', 'fig_gpe_partition.png')
    fig.savefig(out, bbox_inches='tight', dpi=220)
    print('written:', out)
    tab = write_table('gpe_partition', rows[0], rows[1:],
                      script='fig_gpe_partition.py',
                      figure='fig_gpe_partition.png', models=('STD', 'WAL'),
                      meta={'zc_km': cpc.ZC_KM,
                            'additivity': 'trench pull + ridge push = Delta GPE*; '
                                          'isostatic part - tilt = ridge push; '
                                          'both asserted at render',
                            'tilt': '|Delta P| * z_c on the analysis grid',
                            'static_limit': 'cpc.partition static (plateau asymptote), '
                                            'reference only — not additive with the rest',
                            'no_residual_panel': 'deliberate; see DYNAMICS_FINDINGS §5.10'})
    print('written:', tab)


if __name__ == '__main__':
    main()
