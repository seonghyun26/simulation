def plot_ramachandran(traj, molecule_name, phi_atoms=None, psi_atoms=None):
    if phi_atoms is None:
        phis = mdtraj.compute_phi(traj)[1].ravel()
    else:
        phis = mdtraj.compute_dihedrals(
            traj, mdtraj.geometry.dihedral._atom_sequence(traj.topology, phi_atoms)[1]
        )
    if psi_atoms is None:
        psis = mdtraj.compute_psi(traj)[1].ravel()
    else:
        psis = mdtraj.compute_dihedrals(
            traj, mdtraj.geometry.dihedral._atom_sequence(traj.topology, psi_atoms)[1]
        )
    fig = plt.figure()
    gs = GridSpec(nrows=2, ncols=3)
    # Ramachandran plot
    ax1 = fig.add_subplot(gs[:2, :2])
    ax1.plot(phis * 180 / np.pi, psis * 180 / np.pi, "k+")
    ax1.set_aspect("equal", adjustable="box")
    ax1.axvline(0)
    ax1.axhline(0)
    ax1.set_xlim(-180, 180)
    ax1.set_ylim(-180, 180)
    ax1.set_xticks(np.linspace(-180, 180, 5))
    ax1.set_yticks(np.linspace(-180, 180, 5))
    ax1.set_xlabel("Phi [deg]")
    ax1.set_ylabel("Psi [deg]")
    # Phi(t) plot
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(np.arange(len(phis)), phis * 180 / np.pi, "k+")
    ax2.axhline(0)
    ax2.set_ylim(-180, 180)
    ax2.set_yticks(np.linspace(-180, 180, 5))
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Phi [deg]")
    # Psi(t) plot
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.plot(np.arange(len(phis)), psis * 180 / np.pi, "k+")
    ax3.axhline(0)
    ax3.set_ylim(-180, 180)
    ax3.set_yticks(np.linspace(-180, 180, 5))
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Psi [deg]")
    fig.tight_layout()
    
    # Save the figure
    plt.savefig(f"{molecule_name}_ram_plot.png")