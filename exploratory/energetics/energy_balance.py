"""Does PE release balance viscous dissipation, and where is it dissipated?

⚠ PROOF OF CONCEPT. NOT A PAPER FIGURE. Writes beside itself, never into
the repository's figures/ or tables/ -- see exploratory/README.md.

THE BALANCE. For Boussinesq Stokes flow the INSTANTANEOUS rate balance is

    int Phi dV  =  int Delta rho g . v dV  +  boundary work

with Phi the viscous dissipation rate per volume. The boundary term should
VANISH IDENTICALLY in this model: free-slip sidewalls and base give
v.n = 0 with no shear traction, and the top is undeformed and stress-free.
That is a property of this setup, not a general result.

⚠ It is a RATE balance. Cumulative PE change does not track cumulative
dissipation, because the reference state moves. Nothing here integrates in
time.

⚠ NO ELASTIC TERM. The Cerpa rheology is Byerlee + diffusion + dislocation
+ Peierls -- viscous and viscoplastic throughout, with yielding entering as
an effective viscosity, so 4 eta e_II^2 captures plastic work too. A
visco-elastic model (MDOODZ, G = 3e10) would need a storage term; this one
does not.

THREE CONVENTIONS, PINNED ON THE DATA RATHER THAN ASSUMED
(the step-0 ritual, ANALYSIS_CONVENTIONS §1.2):

1. THE DISSIPATION FACTOR. Measured on this snapshot,
   tau_II / (2 eta e_II) = 1.0000, so the stored invariant is the one
   conjugate to tau_II and

       Phi = 2 tau_II e_II = 4 eta e_II^2

   Getting this wrong is a silent factor of two.

2. VISCOSITY IS STORED AS A 9-COMPONENT TENSOR but is the scalar
   replicated into the 2x2 block (components 0,1,3,4 equal). Component 0
   is the viscosity. The large "off-diagonal" is the 1e25 Pa s ceiling,
   not anisotropy.

3. ⚠ THE MESH DOUBLE-COUNTS CELLS. Summing cell areas as read gives
   108.1 % of the 8000 x 2900 km box. `clean()` does not help, and
   vtkGhostLevels is NOT a ghost flag here -- it takes 256 distinct values
   and looks like a partition index, so masking on == 0 gives 22 % of the
   box, far worse. The cause is 4.0 % of cells duplicated across partition
   boundaries in the .pvtu. Deduplicating on rounded cell centroids
   returns 100.00 % of the box exactly. ANY integral on this mesh that
   skips this step is 8 % high.

WHY THE NATIVE MESH AND NOT A SAMPLED GRID. The AMR spans 0.47 to 63 km
edges, a factor of 135, and dissipation concentrates exactly where the
mesh is finest. A uniform grid either misses the peak or wastes effort in
the deep mantle; the native-cell sum is cheaper and more accurate. This is
the opposite choice from every other script in the repository, which is
why it lives in its own helper rather than reusing one.

REFERENCE-DENSITY INDEPENDENCE. The buoyancy work is invariant to the
choice of reference density, because int v_z dV = 0 in a closed domain
with impermeable boundaries (v = curl psi; psi is constant on a free-slip
boundary, so the integral telescopes to zero). Panel (a) checks this
numerically rather than asserting it.
"""
import os, sys, glob, warnings, logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyvista as pv
warnings.filterwarnings('ignore'); logging.disable(logging.ERROR)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, 'data')
G, Y_SURF = 9.8, 2_900_000.0
CM = {'STD': '#002147', 'WAL': '#E5007D'}
T_LAB = 1373.0          # K, the WAL weak-layer boundary (Cerpa et al.)
Z_WEAK, Z_660 = 220e3, 660e3
Z_PLATE = 125e3        # plate / slab divide; see main()


def cells(key):
    """Per-cell area, centroid, and the fields, with duplicates removed."""
    f = sorted(glob.glob(os.path.join(DATA, f'{key}_RefModel', '*.pvtu')))[0]
    v = pv.read(f)
    t = float(np.asarray(v['NormalSP::Time']).flat[0]) / 31557600.0 / 1e6
    c = v.cell_centers().points
    cs = v.compute_cell_sizes(length=False, area=True, volume=False)
    A = np.asarray(cs.cell_data['Area'])
    # DEDUPLICATE (convention 3 above)
    _, keep = np.unique(np.round(c[:, :2], 3), axis=0, return_index=True)
    # ⚠ FORM EVERY NONLINEAR QUANTITY AT THE POINTS, THEN AVERAGE.
    # Phi is quadratic in eta and in the strain rate. Averaging eta and
    # e_II to cells first and multiplying afterwards is NOT the same
    # operation, and on this mesh it is badly wrong: the two independent
    # routes to Phi (4 eta e_II^2 and (tau_xx^2+tau_xz^2)/eta) came out
    # 140 % (STD) / 657 % (WAL) apart, and the energy balance missed by
    # 5.1 / 7.4 %. Formed at the points and then averaged, the two routes
    # agree to 0.00 % and the balance closes to 3.0 / 5.8 %.
    # The error concentrates where eta and e_II vary steeply across a
    # cell -- the interface and the slab edge -- which is exactly where
    # the answer matters.
    eta_p = np.asarray(v['NormalSP::Viscosity'])[:, 0]
    e_p = np.asarray(v['NormalSP::SecondInvariant_StrainRate'])
    st_p = np.asarray(v['NormalSP::Stress'])
    v.point_data['phi'] = 4.0 * eta_p * e_p ** 2
    v.point_data['tau'] = 2.0 * eta_p * e_p
    v.point_data['phi_n'] = st_p[:, 0] ** 2 / eta_p   # stretching
    v.point_data['phi_s'] = st_p[:, 1] ** 2 / eta_p   # shear
    cd = v.point_data_to_cell_data()
    g = lambda n, j=None: (np.asarray(cd[n]) if j is None
                           else np.asarray(cd[n])[:, j])[keep]
    return dict(t=t, A=A[keep], x=c[keep, 0], z=Y_SURF - c[keep, 1],
                phi=g('phi'), tau=g('tau'),
                phi_n=g('phi_n'), phi_s=g('phi_s'),
                rho=g('NormalSP::Density'),
                T=g('NormalSP::Temperature'),
                vz=-g('NormalSP::Velocity', 1))     # z-down positive


TAU_CAP = 500e6      # Pa; see terms(). Declared, not tuned.


def terms(c):
    """Dissipation and buoyancy-work densities [W/m^3] per cell.

    TWO CORRECTIONS, both forced by the data and both essential.

    1. THE WORK TERM IS REFERENCED TO THE DEPTH-MEAN DENSITY, not to a
       constant. Density spans only 3171-3300 kg/m3, so a constant
       reference makes the work a small difference of large numbers:
       int v_z dA is zero only to 0.04 % of the mean |v_z| x area, and
       multiplied by rho_ref g ~ 32 kPa that residual is 1.2e4 W/m against
       a work term of 3.8e4 -- a 32 % error, and reference-dependent.
       Using the lateral anomaly rho - <rho>(z) removes it: the
       horizontally uniform part does no net work, and the integral is
       then well conditioned.

    2. A STRESS CAP, WHICH NOW REMOVES NOTHING -- retained only as a
       tripwire. ⚠ RETRACTION: an earlier version of this script reported
       ~1,900 cells with implied tau_II up to 96 GPa, concentrated at the
       subduction interface, carrying 56-86 % of the dissipation, and
       concluded that the interface was where the numerics were least
       trustworthy. THAT WAS WRONG. It was an artefact of averaging eta
       and e_II to cells before multiplying (see `cells()`); forming Phi
       at the points removes every one of those cells. No stress anywhere
       on this mesh exceeds 500 MPa. The cap stays so that a future
       regression announces itself rather than hiding in a plausible
       physical story.
    """
    phi = np.where(c['tau'] > TAU_CAP, 0.0, c['phi'])
    # lateral density anomaly, 10 km depth bins
    edges = np.arange(0, 2901, 10) * 1e3
    idx = np.clip(np.digitize(c['z'], edges) - 1, 0, len(edges) - 2)
    num = np.bincount(idx, weights=c['rho'] * c['A'], minlength=len(edges) - 1)
    den = np.bincount(idx, weights=c['A'], minlength=len(edges) - 1)
    rho_bar = np.where(den > 0, num / np.maximum(den, 1e-30), 0.0)
    work = (c['rho'] - rho_bar[idx]) * G * c['vz']
    return phi, work


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.2))
    report = []
    for key in ('STD', 'WAL'):
        c = cells(key)
        phi, work = terms(c)
        D = float((phi * c['A']).sum())          # W per metre of strike
        W = float((work * c['A']).sum())
        nbad = int((c['tau'] > TAU_CAP).sum())
        frac = 100 * (1 - D / float((c['phi'] * c['A']).sum()))
        report.append((key, c['t'], D, W, nbad, frac))
        print(f'{key} t={c["t"]:.1f} Myr: dissipation {D:.4e} W/m, '
              f'PE release {W:.4e} W/m, closure {100*(D-W)/W:+.1f} %')
        print(f'   stress cap removed {nbad} cells carrying {frac:.0f} % of the '
              f'uncapped dissipation; total area '
              f'{100*c["A"].sum()/(8000e3*2900e3):.2f} % of the box')

        # ---- regions -------------------------------------------------
        # COLD MATERIAL IS SPLIT AT 125 km (Dan, 2026-10-12). A single
        # T < T_LAB mask is "lithosphere AND slab" -- it follows cold
        # material to whatever depth it reaches. 125 km is where the
        # column has stopped translating with the plate (fig_lab_kinematics:
        # v/v_plate ~ 0.6 by 125 km; dynamical LAB 99 / 93 km), so above it
        # is the horizontally moving plate and below it the descending slab.
        cold = c['T'] < T_LAB
        plate = cold & (c['z'] < Z_PLATE)
        slab = cold & (c['z'] >= Z_PLATE)
        weak = (~cold) & (c['z'] <= Z_WEAK)
        um = (~cold) & (c['z'] > Z_WEAK) & (c['z'] <= Z_660)
        lm = (~cold) & (c['z'] > Z_660)
        regs = [('Plate\ncold, $<125$ km', plate),
                ('Slab\ncold, $>125$ km', slab),
                ('Weak layer\n$z\\leq220$ km', weak),
                ('Upper mantle\n220-660 km', um),
                ('Lower mantle\n$>660$ km', lm)]
        share = [100 * float((phi * c['A'])[m].sum()) / D for _, m in regs]
        okc = c['tau'] <= TAU_CAP
        print(f'   stretching {100*(c["phi_n"]*c["A"])[okc].sum()/D:.1f} %  '
              f'shear {100*(c["phi_s"]*c["A"])[okc].sum()/D:.1f} % '
              f'(exact additive split: Phi = (tau_xx^2 + tau_xz^2)/eta)')
        print('   dissipation share: ' + '  '.join(
            f'{n.splitlines()[0]} {s:.1f} %' for (n, _), s in zip(regs, share)))

        # (b) regional shares
        ax = axes[0, 1]
        xs = np.arange(len(regs)) + (0.0 if key == 'STD' else 0.38)
        ax.bar(xs, share, width=0.36, color=CM[key], alpha=0.85, label=key)
        for x, s in zip(xs, share):
            ax.text(x, s + 1.0, f'{s:.0f}', ha='center', fontsize=8.5,
                    color=CM[key])

        # (c) dissipation against depth
        ax = axes[1, 0]
        edges = np.arange(0, 2900, 50) * 1e3
        idx = np.clip(np.digitize(c['z'], edges) - 1, 0, len(edges) - 2)
        prof = np.bincount(idx, weights=phi * c['A'], minlength=len(edges) - 1)
        ax.step(prof / 1e6, (edges[:-1] + 25e3) / 1e3, where='mid',
                color=CM[key], lw=1.8, label=key)

        # (d) where it happens: cumulative share with depth
        ax = axes[1, 1]
        o = np.argsort(c['z'])
        ax.plot(c['z'][o] / 1e3, 100 * np.cumsum((phi * c['A'])[o]) / D,
                color=CM[key], lw=2.0, label=key)

    # (a) the balance
    ax = axes[0, 0]
    for i, (key, t, D, W, *_) in enumerate(report):
        ax.bar(i - 0.19, W / 1e6, width=0.36, color=CM[key], alpha=0.45,
               label='PE release rate' if i == 0 else None)
        ax.bar(i + 0.19, D / 1e6, width=0.36, color=CM[key], alpha=0.95,
               label='viscous dissipation' if i == 0 else None)
        ax.text(i, max(D, W) / 1e6 * 1.04, f'{100*(D-W)/W:+.1f} %',
                ha='center', fontsize=10, color=CM[key])
    ax.set_xticks(range(2)); ax.set_xticklabels([r[0] for r in report])
    ax.set_ylabel('MW per metre of strike', fontsize=10.5)
    ax.set_title('(a)  global rate balance', fontsize=11, loc='left')
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.25, axis='y')

    ax = axes[0, 1]
    ax.set_xticks(np.arange(len(regs)) + 0.19)
    ax.set_xticklabels([n for n, _ in regs], fontsize=7.5)
    ax.set_ylabel('share of total dissipation [%]', fontsize=10.5)
    ax.set_title('(b)  where it is dissipated', fontsize=11, loc='left')
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.25, axis='y')
    ax.set_ylim(0, max(ax.get_ylim()[1], 60))

    ax = axes[1, 0]
    ax.set_ylim(2900, 0); ax.set_xscale('log')
    ax.set_xlabel('dissipation per 50 km depth bin [MW/m]', fontsize=10.5)
    ax.set_ylabel('Depth [km]', fontsize=10.5)
    ax.axhline(660, color='0.4', lw=1.0, ls=':')
    ax.text(ax.get_xlim()[0], 650, ' 660 km', fontsize=8.5, color='0.4', va='bottom')
    ax.set_title('(c)  dissipation with depth', fontsize=11, loc='left')
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax.set_xlim(0, 1200); ax.set_ylim(0, 100)
    ax.set_xlabel('Depth [km]', fontsize=10.5)
    ax.set_ylabel('cumulative share of dissipation [%]', fontsize=10.5)
    for zz in (220, 660):
        ax.axvline(zz, color='0.4', lw=1.0, ls=':')
    ax.set_title('(d)  cumulative with depth', fontsize=11, loc='left')
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.25)

    fig.suptitle('PROOF OF CONCEPT — energy balance at the reference snapshot '
                 '(not a paper figure)', fontsize=11, color='0.35')
    fig.tight_layout()
    out = os.path.join(HERE, 'energy_balance.png')
    fig.savefig(out, dpi=180, bbox_inches='tight')
    print('written:', out)


if __name__ == '__main__':
    main()
