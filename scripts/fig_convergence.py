"""fig_convergence — over what depth is the trench pressure deficit equilibrated?

Writes figures/fig_convergence.png and tables/convergence.csv from the
committed column-profile cache.

SI FIGURE. It asks one question, which is the use h_m is put to in the
manuscript: is the mechanical thickness a fair proxy for the depth over
which the trench pressure deficit equilibrates.

WHAT IT REPLACES (2026-10-11). The figure carried five curves -- Delta N_D,
both Delta GPE* terms, M and V -- under the single heading "convergence".
They do not behave alike and the heading hid it: measured as the fraction
reached by h_m, M gets to only 87.5 / 88.5 % (its first-moment weighting
keeps it accruing below the plate) and N_D(x_T) has no well-defined
fraction at all, because its two lobes cancel and the running integral
passes through zero. Those belong in their own figures, not averaged into
one claim.


Dan, 2026-10-11. The figure now asks one question -- over what depth is the
trench pressure deficit equilibrated, and is h_m a fair proxy for it -- which
is the use h_m is put to in the manuscript. The ridge-side term is dropped:
its depth structure is already carried by fig_ridge_cumulative and
fig_ridge_push_anatomy, and it brought a problem with it (see below).

WHY THE RIDGE SIDE IS NOT HERE. Referenced to z_c, the ridge-side cumulative
at h_m exceeds 100 % in the early snapshots -- 139 % at t = 14 Myr in STD.
That is arithmetic, not a bug: the ridge-side anomaly changes sign at z*,
and while z* sits ABOVE h_m the cumulative has already peaked and is falling
by the time it reaches h_m, so it holds more there than at z_c. z* is that
shallow early because the ridge column is still at the model wall -- the
fluidity ridge is emergent and migrates inward (conventions §3.4), and x_R
runs 70 -> 476 km (STD) / 222 -> 732 km (WAL) across the run. Every
over-100 % snapshot is one with z* above h_m, and they stop once the ridge
clears the corner: t >= 38 Myr (STD), t >= 26 Myr (WAL).
⚠ That is worth pursuing separately: every ridge-referenced run statistic
currently includes those wall-adjacent snapshots, and they are past the
8 Myr spin-up cut, so nothing excludes them.
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import column_profiles_cache as cpc
from fig_mechanical_thickness import thicknesses, YIELD_CAP_MPA, FLEX_COL
from tables_io import write_table

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C={'STD':'#002147','WAL':'#E5007D'}; C_RULE='#BFC3D1'; C_T='b'
ZC=cpc.ZC_KM; Z_MAX=120.0
d=cpc.load(); sm=lambda a: gaussian_filter1d(a,2)

rows=[('model','quantity','value')]
fig=plt.figure(figsize=(10.2,9.2))
gs=fig.add_gridspec(2,2,height_ratios=[1.45,1.0],hspace=0.30,wspace=0.08)
top=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1])]
top[1].sharey(top[0])

for i,key in enumerate(('STD','WAL')):
    ax=top[i]; c=cpc.derive(d,key); z=c['z']; m=c['mid']
    hm=np.nanmedian(thicknesses(d,key,col='T')['yield_cap'][m])/1e3
    prof=-(d[f'{key}_szz_I']-d[f'{key}_szz_T'])
    p=np.mean([sm(prof[j]) for j in np.where(m)[0]],axis=0)
    ze,cu=cpc.cumulative_edges(p,z); zk=ze/1e3
    ref=float(np.interp(ZC,zk,cu))
    ax.plot(cu/ref, zk, '-', color=C_T, lw=2.8, alpha=0.9,
            label=r'$\Delta\mathrm{GPE}^{*}$, trench side')
    # Uniform elastic plate: a parabolic equivalent density through the
    # thickness gives Delta sigma_zz = Delta P_T (1 - 3 zeta^2 + 2 zeta^3)
    # and a cumulative 2 zeta - 2 zeta^3 + zeta^4, zeta = z/h_m. It
    # integrates to (1/2) Delta P_T h_m, so it carries a_eff = h_m/2 by
    # construction -- the ARM is therefore not what it tests. The SHAPE is.
    zet=np.linspace(0,1,200)
    ax.plot(2*zet-2*zet**3+zet**4, zet*hm, ls=(0,(5,2)), color=C_T, lw=1.6,
            zorder=4, label='Uniform elastic plate (prediction)')
    ax.plot([1,1],[hm,Z_MAX],ls=(0,(5,2)),color=C_T,lw=1.6,zorder=4)
    at_h=float(np.interp(hm,zk,cu))/ref
    ax.plot(at_h,hm,'o',color=C_T,ms=7,mec='white',mew=1.2,zorder=6)
    ax.axhline(hm,color=C[key],lw=1.6,ls=(0,(6,3)))
    ax.axhline(ZC,color='0.45',lw=1.1,ls=':')
    ax.text(1.13,hm-1.2,f'$h_m$ = {hm:.0f} km',fontsize=9,color=C[key],
            ha='right',va='bottom')
    ax.text(1.13,ZC+1.2,f'$z_c$ = {ZC:.0f} km',fontsize=9,color='0.45',
            ha='right',va='top')
    ax.axvline(1,color='k',lw=0.9)
    ax.set_ylim(Z_MAX,0); ax.set_xlim(0,1.15)
    ax.set_xlabel(r'Cumulative $\Delta\mathrm{GPE}^{*}(z)$, normalised at $z_c$',
                  fontsize=10.5)
    ax.set_title(f'({"ab"[i]})  {key}',fontsize=11,loc='left')
    ax.grid(alpha=0.25,color=C_RULE,lw=0.6)
    if i: ax.tick_params(labelleft=False)
    zz=np.linspace(0.02,1,50); mod=2*zz-2*zz**3+zz**4
    dat=np.interp(zz*hm,zk,cu)/ref
    rms=float(np.sqrt(np.mean((dat-mod)**2)))
    rows += [(key,'midrun_h_m_km',f'{hm:.1f}'),
             (key,'midrun_pct_of_zc_reached_by_h_m',f'{100*at_h:.1f}'),
             (key,'elastic_misfit_rms',f'{rms:.3f}'),
             (key,'elastic_max_dev',f'{np.max(np.abs(dat-mod)):.3f}'),
             (key,'elastic_max_dev_at_z_over_h_m',
              f'{zz[int(np.argmax(np.abs(dat-mod)))]:.2f}')]
    print(f'{key}: at h_m {100*at_h:.1f} % of the z_c value; '
          f'elastic misfit rms {rms:.3f}')
top[1].tick_params(labelleft=False)
top[0].set_ylabel('Depth [km]',fontsize=11)
top[0].legend(frameon=False,fontsize=9.5,loc='lower left')

for i,key in enumerate(('STD','WAL')):
    ax=fig.add_subplot(gs[1,i])
    c=cpc.derive(d,key); z=c['z']; t=c['t']
    hmv=thicknesses(d,key,col='T')['yield_cap']
    prof=-(d[f'{key}_szz_I']-d[f'{key}_szz_T'])
    F=[]
    for j in range(len(t)):
        ze,cu=cpc.cumulative_edges(sm(prof[j]),z)
        F.append(cu[np.argmin(np.abs(ze-hmv[j]))]
                 /cu[np.argmin(np.abs(ze-ZC*1e3))])
    F=100*np.array(F)
    ax.plot(t,F,'-',color=C_T,lw=2.4,alpha=0.9)
    ax.axhline(95,color='0.35',lw=1.1,ls=(0,(5,3)))
    ax.text(79,95.3,'95 %',fontsize=9,color='0.35',ha='right',va='bottom')
    ax.set_ylim(93,101); ax.grid(alpha=0.25,color=C_RULE,lw=0.6)
    ax.set_xlabel('Model time [Myr]',fontsize=11)
    ax.set_title(f'({"cd"[i]})  {key}',fontsize=11,loc='left')
    if i: ax.tick_params(labelleft=False)
    else: ax.set_ylabel('Cumulative $\\Delta\\mathrm{GPE}^{*}$ reached by $h_m$\n'
                        '[% of its value at $z_c$]',fontsize=10.5)
    rows += [(key,'pct_of_zc_reached_by_h_m_median',f'{np.median(F):.1f}'),
             (key,'pct_of_zc_reached_by_h_m_min',f'{F.min():.1f}'),
             (key,'pct_of_zc_reached_by_h_m_max',f'{F.max():.1f}'),
             (key,'n_snapshots',f'{len(t)}')]
    print(f'{key}: reached by h_m, median {np.median(F):.1f} %, '
          f'range {F.min():.1f}-{F.max():.1f} %')
fig.suptitle('Equilibration depth of the trench pressure deficit',
             fontsize=11,color='0.35')
out=os.path.join(ROOT,'figures','fig_convergence.png')
fig.savefig(out,dpi=200,bbox_inches='tight'); print('written:',out)
print('written:', write_table(
    'convergence', rows[0], rows[1:], script='fig_convergence.py',
    figure='fig_convergence.png', models=('STD','WAL'),
    meta={'quantity':'trench-side Delta GPE*, cumulative with depth',
          'normalisation':'each curve by its own value at z_c',
          'zc_km':ZC, 't_min_Myr':cpc.T_MIN_MYR,
          'h_m':f'{YIELD_CAP_MPA:.0f} MPa carried fibre stress, '
                f'flex column {FLEX_COL}',
          'h_m_in_panels_ab':'mid-run median; panels c,d use the '
                             'per-snapshot h_m',
          'midrun_window_Myr':list(cpc.MIDRUN_MYR),
          'elastic_model':'parabolic equivalent density over 0..h_m; '
                          'cumulative 2z-2z^3+z^4 in z/h_m; carries '
                          'a_eff = h_m/2 BY CONSTRUCTION, so it tests the '
                          'SHAPE and not the arm',
          'ridge_side':'EXCLUDED 2026-10-11 -- its depth structure is in '
                       'fig_ridge_cumulative and fig_ridge_push_anatomy'}))
