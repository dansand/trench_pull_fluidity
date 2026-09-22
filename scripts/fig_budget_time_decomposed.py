"""fig_budget_time_decomposed — the budget through time with the driving term split.

Writes figures/fig_budget_time_decomposed.png and
tables/budget_time_decomposed.csv from the committed column-profile cache.

Requested by Dan 2026-09-23. fig_budget_time shows that the
short-wavelength oscillation of the budget sits in Delta N_D and F_B
while the GPE-like term looks comparatively smooth. His question: is
that partly an artefact of CANCELLATION -- the trench pull and the ridge
push oscillating against each other inside the single blue curve, with
the tilt as the mechanism that ties them?

This figure is fig_budget_time with the driving term opened up:

    -Delta GPE*  =  -(trench pull)  +  -(ridge push)
                     non-isostatic      isostatic
                     x_T -> x_I         x_I -> x_R

and the ridge push itself carries the tilt,

    ridge push = non-tilting (thermal) - tilting

so the tilting contribution is drawn as its own curve. Note that the
non-tilting term is DEFINED as ridge push + tilting, so those two are not
independent measurements: the pair is a decomposition, and the figure
shows how large each part is, not that one causes the other.

Sign convention is fig_budget_time's, unchanged: every curve is that
term's contribution to the net force in +x, so

    negative = force toward the trench = DRIVING
    positive = RESISTING

Colours are the schematic's two domains (Dan, 2026-09-23): the
non-isostatic / trench pull domain blue #0072B2 and the isostatic /
ridge push domain orange #D55E00, matching fig_column_anomalies and
fig_gpe_decomposition. Tilting keeps the purple used for it in the
kinematics figures; Delta N_D black, F_B red, closure green dashed as
before.

  top     the balance terms through time, decomposed, with each secular
          trend drawn thin.
  bottom  the residual about those trends -- where the oscillation Dan is
          asking about actually lives -- with the domain curves and the
          two response terms on one axis.

THE ANSWER: DAN'S SUSPICION IS RIGHT, AND ONLY PARTLY SUFFICIENT.

  residual s.d. [TN/m]              STD     WAL
    trench pull                     0.145   0.270
    ridge push                      0.103   0.179
    ---- if they were independent   0.178   0.324   (quadrature)
    ---- if they moved together     0.248   0.449   (coherent sum)
    their sum, MEASURED             0.110   0.259
    tilting                         0.099   0.106
    Delta N_D                       0.289   0.319
    F_B                             0.309   0.243

The two domains DO oppose each other: r = -0.65 (STD) / -0.40 (WAL) on
the detrended series. The measured sum is 38 % (STD) / 20 % (WAL) below
what independent fluctuations would give, so real cancellation is hiding
inside the single blue curve of fig_budget_time, and the trench pull on
its own fluctuates MORE than the whole driving term does.

But cancellation does not account for the whole contrast. Even with the
opposition switched off, the drive would fluctuate at 0.178 (STD) against
0.289 for Delta N_D and 0.309 for F_B. Cancellation closes roughly half
the gap; the other half is that each domain is genuinely quieter than
either response term. So: the oscillation really is concentrated in
Delta N_D and F_B, but the margin is about half what fig_budget_time
alone suggests, and that figure should not be read as showing a steady
drive.

THE MECHANISM DIFFERS BETWEEN THE RUNS, which is the channel dial doing
what it should. All correlations below are on detrended magnitudes.

  STD  the opposition is carried by the TILT. Trench pull tracks tilting
       at +0.55 and the thermal part at only -0.12, while ridge push
       tracks tilting at -0.45. A faster plate deepens the trench and
       simultaneously tilts the isostatic domain down toward the ridge,
       and the two effects subtract in the total.

  WAL  the tilt route is weak -- trench pull against tilting is only
       +0.10 -- and the opposition sits instead between the trench pull
       and the THERMAL part, -0.35. With the weak asthenospheric layer
       the channel gradient is small, so the cancellation is both weaker
       (20 %) and differently sourced.

⚠ The trench pull carries most of the drive's own fluctuation in both
runs, 58 % (STD) / 60 % (WAL) of the summed domain residual s.d., despite
the ridge push being the larger part of the SECULAR rise
(fig_gpe_decomposition: 64 % / 72 %). The two domains divide the labour
by timescale.

The relevant earlier finding is recorded in DYNAMICS_FINDINGS §1: the
short-wavelength coupling is slab-driven and arrives at the plate as a
relaxation of Delta N_D (STD), with the drive's TOTAL barely moving. This
figure keeps that conclusion but sharpens it -- the total barely moves
because its two parts move and cancel, not because nothing is happening
inside it.

DRAFT CAPTION. The force budget through time with the driving term
decomposed, for STD (left) and WAL (right). Top: the trench pull across
the non-isostatic domain (blue) and the ridge push across the isostatic
domain (orange), which sum to the GPE-like driving force (grey), against
the normal-stress-difference resultant (black) and the accumulated basal
traction (red); their sum is the closure (green dashed). The tilting
contribution carried inside the ridge push is shown in purple, and the
ridge push it would supply without the tilt as the dashed orange line.
Curves are signed so that negative acts toward the trench and drives the
plate. Thin lines are the fitted secular trends. Bottom: the residual
about those trends, with each series' standard deviation. The two domains
fluctuate in opposition (r = -0.65 in STD, -0.40 in WAL), so their sum
fluctuates 38 % and 20 % less than independent parts would, and the
trench pull alone is more variable than the whole driving term. The
opposition is mediated by the tilt in STD and by the thermal part in WAL.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import column_profiles_cache as cpc
from fig_budget_time import terms, T_MIN_MYR
from tables_io import write_table

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C_RULE = '#BFC3D1'
C_TRENCH = '#0072B2'        # non-isostatic / trench pull domain (schematic)
C_RIDGE = '#D55E00'         # isostatic / ridge push domain (schematic)
C_TILT = '#7B3294'
C_TOTAL = '0.55'


def main():
    if not hasattr(np, 'trapezoid'):
        np.trapezoid = np.trapz
    d = cpc.load()
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.4), sharex=True,
                             gridspec_kw={'height_ratios': [1.6, 1.0]})
    rows = [('model', 'quantity', 'value')]
    ld = lambda t, y: y - np.polyval(np.polyfit(t, y, 1), t)
    fit = lambda t, y: np.polyval(np.polyfit(t, y, 1), t)

    for col, key in enumerate(('STD', 'WAL')):
        c = cpc.derive(d, key)
        q = terms(d, key)
        z, zc = c['z'], c['zc']
        m = q['t'] >= T_MIN_MYR
        t = q['t'][m]

        # the driving term, opened up. TP + RP reproduces terms()' d_gpe
        # exactly (both are the szz difference between the trench and ridge
        # columns over 0..z_c, routed through the x_I anomalies) -- asserted
        # below rather than assumed.
        TP = -np.trapezoid(c['p_T'][:, zc], z[zc], axis=1)[m] / 1e12
        RP = np.trapezoid(c['p_R'][:, zc], z[zc], axis=1)[m] / 1e12
        NT = np.trapezoid(c['p_R_untilted'][:, zc], z[zc], axis=1)[m] / 1e12
        # The tilt contribution |Delta P| * z_c, evaluated ON THE ANALYSIS
        # GRID rather than at the nominal z_c. The cache's z is cell-centred
        # (500 m .. 74,500 m), so the column integral spans 74 km, and using
        # the nominal 75 km would break the decomposition by 1.3 % --
        # 0.010 TN/m against a ~0.7 TN/m tilt. Same quantity, same
        # convention (§6b); the identity ridge push = non-tilting - tilting
        # then holds to machine precision, which is what lets this figure
        # claim to be a decomposition rather than a comparison.
        TILT = -c['dP'][m] * (z[zc][-1] - z[zc][0]) / 1e12
        GPE, ND, FB = (q['d_gpe'][m] / 1e12, q['d_nd'][m] / 1e12,
                       q['f_b'][m] / 1e12)
        CLO = q['closure'][m] / 1e12
        assert np.allclose(TP + RP, GPE, atol=2e-3), \
            f'{key}: TP + RP does not reproduce the GPE-like term'
        assert np.allclose(NT - TILT, RP, atol=2e-3), \
            f'{key}: non-tilting - tilting does not reproduce the ridge push'

        # everything in the budget sign convention: contribution to +x
        series = [('trench pull', -TP, C_TRENCH, 2.0, '-'),
                  ('ridge push', -RP, C_RIDGE, 2.0, '-'),
                  ('non-tilting (thermal)', -NT, C_RIDGE, 1.1, '--'),
                  ('tilting', TILT, C_TILT, 1.4, '-')]

        a0, a1 = axes[0, col], axes[1, col]
        a0.axhline(0, color='k', lw=1.6)
        a0.plot(t, -GPE, color=C_TOTAL, lw=4.5, alpha=0.55,
                label=r'$-\Delta\mathrm{GPE}^{*}$ (total)')
        for nm, y, c_, lw, ls in series:
            a0.plot(t, y, color=c_, lw=lw, ls=ls, label=nm)
            a0.plot(t, fit(t, y), color=c_, lw=0.8, alpha=0.85)
        a0.plot(t, ND, color='k', lw=1.6, label=r'$\Delta N_D$')
        a0.plot(t, FB, color='red', lw=2.0, label=r'$F_B$')
        a0.plot(t, CLO, color='g', ls='--', lw=2.2, label='closure')
        for y in (ND, FB):
            a0.plot(t, fit(t, y), color='k' if y is ND else 'red',
                    lw=0.8, alpha=0.85)
        a0.set_title(key, fontsize=11)
        a0.grid(alpha=0.25, color=C_RULE, lw=0.6)

        # --- residuals: where the oscillation Dan asked about lives -----
        res = {nm: ld(t, y) for nm, y, *_ in series}
        res['drive (TP + RP)'] = ld(t, -GPE)
        res[r'$\Delta N_D$'] = ld(t, ND)
        res['$F_B$'] = ld(t, FB)
        # CORRELATIONS ARE COMPUTED ON THE PHYSICAL MAGNITUDES (trench pull,
        # ridge push, tilting and the thermal part all taken positive), NOT
        # on the plotted curves, which carry the budget's +x sign flip. A
        # negative r below therefore means the two quantities oppose each
        # other, in the ordinary reading, and the table needs no sign key.
        # The three ridge quantities are not independent -- non-tilting =
        # ridge push + tilting -- so these say which part the ridge push
        # follows, not which one causes it.
        rTR = float(np.corrcoef(ld(t, TP), ld(t, RP))[0, 1])
        rRT = float(np.corrcoef(ld(t, RP), ld(t, TILT))[0, 1])
        rTT = float(np.corrcoef(ld(t, TP), ld(t, TILT))[0, 1])
        rTN = float(np.corrcoef(ld(t, TP), ld(t, NT))[0, 1])
        quad = np.hypot(res['trench pull'].std(), res['ridge push'].std())
        coh = res['trench pull'].std() + res['ridge push'].std()
        canc = 100 * (1 - res['drive (TP + RP)'].std() / quad)
        tshare = 100 * res['trench pull'].std() / coh

        for nm, c_, lw, ls in (('trench pull', C_TRENCH, 1.8, '-'),
                               ('ridge push', C_RIDGE, 1.8, '-'),
                               ('tilting', C_TILT, 1.2, '-'),
                               ('drive (TP + RP)', C_TOTAL, 3.0, '-'),
                               (r'$\Delta N_D$', 'k', 1.5, '-'),
                               ('$F_B$', 'red', 1.5, '-')):
            a1.plot(t, res[nm], color=c_, lw=lw, ls=ls,
                    alpha=0.55 if nm == 'drive (TP + RP)' else 1.0,
                    label=f'{nm}  (s.d. {res[nm].std():.3f})')
        a1.axhline(0, color='k', lw=0.8)
        a1.set_xlabel('Model time [Myr]', fontsize=11)
        a1.grid(alpha=0.25, color=C_RULE, lw=0.6)
        a1.legend(frameon=False, fontsize=7.5, loc='upper left', ncol=3)
        a1.set_title(f'residual about the trends — trench pull vs ridge push '
                     f'$r$ = {rTR:+.2f}, cancellation {canc:.0f} %',
                     fontsize=9, color='0.3')

        print(f'{key}: residual s.d. TP {res["trench pull"].std():.3f}, '
              f'RP {res["ridge push"].std():.3f}, sum '
              f'{res["drive (TP + RP)"].std():.3f} (quadrature {quad:.3f}, '
              f'coherent {coh:.3f}), tilt {res["tilting"].std():.3f}, '
              f'dN_D {res[r"$\Delta N_D$"].std():.3f}, '
              f'F_B {res["$F_B$"].std():.3f}; r(TP,RP) {rTR:+.2f}; '
              f'cancellation {canc:.1f} %; TP carries {tshare:.0f} % of the '
              f'summed domain residual')
        print(f'   {key}: r(RP,tilt) {rRT:+.2f}, r(TP,tilt) {rTT:+.2f}, '
              f'r(TP,non-tilting) {rTN:+.2f}; non-tilting residual s.d. '
              f'{res["non-tilting (thermal)"].std():.3f} vs ridge push '
              f'{res["ridge push"].std():.3f}')
        for nm, slug in (('trench pull', 'trench_pull'),
                         ('ridge push', 'ridge_push'),
                         ('non-tilting (thermal)', 'non_tilting'),
                         ('tilting', 'tilting'),
                         ('drive (TP + RP)', 'drive'),
                         (r'$\Delta N_D$', 'delta_nd'), ('$F_B$', 'basal_traction')):
            rows += [(key, f'residual_sd_{slug}_TNm', f'{res[nm].std():.4f}')]
        rows += [(key, 'corr_residual_trench_pull_ridge_push', f'{rTR:.3f}'),
                 (key, 'corr_residual_ridge_push_tilting', f'{rRT:.3f}'),
                 (key, 'corr_residual_trench_pull_tilting', f'{rTT:.3f}'),
                 (key, 'corr_residual_trench_pull_non_tilting', f'{rTN:.3f}'),
                 (key, 'residual_sd_quadrature_TNm', f'{quad:.4f}'),
                 (key, 'residual_sd_coherent_TNm', f'{coh:.4f}'),
                 (key, 'cancellation_percent_vs_quadrature', f'{canc:.1f}'),
                 (key, 'trench_share_of_domain_residual_percent', f'{tshare:.1f}'),
                 (key, 'median_trench_pull_TNm', f'{np.median(TP):.3f}'),
                 (key, 'median_ridge_push_TNm', f'{np.median(RP):.3f}'),
                 (key, 'median_non_tilting_TNm', f'{np.median(NT):.3f}'),
                 (key, 'median_tilting_TNm', f'{np.median(TILT):.3f}')]

    axes[0, 0].set_ylabel('Force per unit distance [TN/m]', fontsize=11)
    axes[1, 0].set_ylabel('residual about\nthe trend [TN/m]', fontsize=9.5)
    axes[0, 0].legend(frameon=False, fontsize=8.5, loc='lower left', ncol=2)
    lo, hi = axes[0, 0].get_ylim()
    axes[0, 0].set_ylim(lo - 0.32 * abs(lo), hi + 0.22 * abs(hi))
    for ax in axes[0, :]:
        for lab, dy, va in (('RESISTING', 0.975, 'top'), ('DRIVING', 0.025, 'bottom')):
            # off-centre so the legend in the left panel does not run into it
            ax.text(0.74, dy, lab, transform=ax.transAxes, ha='center', va=va,
                    fontsize=15, fontweight='bold', color='0.78', zorder=0)
    fig.suptitle('The budget through time with the driving term decomposed into '
                 'the two schematic domains', fontsize=11.5)
    fig.tight_layout()
    out = os.path.join(ROOT, 'figures', 'fig_budget_time_decomposed.png')
    fig.savefig(out, bbox_inches='tight', dpi=220)
    print('written:', out)
    tab = write_table('budget_time_decomposed', rows[0], rows[1:],
                      script='fig_budget_time_decomposed.py',
                      figure='fig_budget_time_decomposed.png',
                      models=('STD', 'WAL'),
                      meta={'zc_km': cpc.ZC_KM, 't_min_Myr': T_MIN_MYR,
                            'sign': 'positive = +x = resisting; negative = driving',
                            'decomposition': 'GPE-like = trench pull + ridge push; '
                                             'ridge push = non-tilting - tilting',
                            'detrend': 'fitted linear model per series'})
    print('written:', tab)


if __name__ == '__main__':
    main()
