import mdtraj
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm

FIG_SIZE = (20, 6)

def plot_ramachandran(loaded_traj, pdb_file, result_path):
    fig = plt.figure()
    fig, ax = plt.subplots(figsize=(6,6))
    phis = mdtraj.compute_phi(loaded_traj)[1].ravel()
    psis = mdtraj.compute_psi(loaded_traj)[1].ravel()
    
    # Start state
    state_traj = mdtraj.load(pdb_file)
    phi_start = mdtraj.compute_phi(state_traj)[1].ravel()
    psi_start = mdtraj.compute_psi(state_traj)[1].ravel()
    ax.scatter(phi_start * 180 / np.pi, psi_start * 180 / np.pi, c='red', s=100, zorder=1)
    
    # Simulation trajectory
    ax.hist2d(phis * 180 / np.pi, psis * 180 / np.pi, 100, norm=LogNorm(), zorder=0)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xticks(np.linspace(-180, 180, 5))
    ax.set_yticks(np.linspace(-180, 180, 5))
    ax.set_xlabel("Phi [deg]")
    ax.set_ylabel("Psi [deg]")
    fig.tight_layout()
    plt.savefig(f"{result_path}/ramachandran.png")
    plt.close(fig)
    
    return fig

def plot_potential_energy(df, result_path):
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot(df['#"Time (ps)"'], df["Potential Energy (kJ/mole)"])
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('Energy (kJ/mol)')
    ax.set_title('Potential energy over time')
    fig.tight_layout()
    plt.savefig(f"{result_path}/potential_energy.png")
    plt.close(fig)
    
    return fig

def plot_total_energy(df, result_path):
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot(df['#"Time (ps)"'], df["Total Energy (kJ/mole)"])
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('Total Energy (kJ/mol)')
    ax.set_title('Total energy over time')
    fig.tight_layout()
    plt.savefig(f"{result_path}/total_energy.png")
    plt.close(fig)
    
    return fig

def plot_temperature(df, result_path):
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot(df['#"Time (ps)"'], df["Temperature (K)"])
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('Temperature (K)')
    ax.set_title('Temperature over time')
    fig.tight_layout()
    plt.savefig(f"{result_path}/temperature.png")
    plt.close(fig)
    
    return fig