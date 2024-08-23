import os
import pandas
import mdtraj
import nglview
import argparse

import numpy as np
import matplotlib.pyplot as plt

from sys import stdout
from matplotlib.gridspec import GridSpec

from util.config import *
from util.simulation import *
from util.plot import plot_ramachandran

os.environ["OPENMM_PLUGIN_DIR"] = "/home/shpark/.conda/envs/cv/lib/python3.9/site-packages/OpenMM.libs/lib"
print(os.environ["OPENMM_PLUGIN_DIR"])

if __name__ == "__main__":
    # Load configs and save them
    print(f"Loading configs...")
    args = init_args()
    log_dir = set_logging(args)
    
    # Load molecule, forcefield, platform, precision and set simluation
    start_pdb = set_molecule(args.molecule, args.state)
    forcefield_files = set_force_field(args.force_field, args.solvent)
    platform, properties = set_platform(args.platform, args.precision)
    simulation = set_simulation(args, forcefield_files, start_pdb, platform, properties)
    
    print(f">> Molecule: {args.molecule}")
    print(f">> Force field: {forcefield_files}")
    print(f">> Time horizon : {args.time}")
    print(f">> Temperature: {args.temperature}")
    print(f">> Platform, precision: {args.platform}, {args.precision}")
    
    
    # Set simulation reporters
    time_horizon = args.time
    traj_file_name = f"{log_dir}/traj.dcd"
    csv_file_name = f"{log_dir}/scalars.csv"
    simulation.reporters = []
    simulation.reporters.append(
        DCDReporter(
            file=traj_file_name,
            reportInterval=args.freq_dcd
    ))
    simulation.reporters.append(
        StateDataReporter(
            stdout,
            reportInterval=args.freq_stdout,
            step=True,
            time=True,
            potentialEnergy=True,
            temperature=True,
            progress=True,
            elapsedTime=True,
            totalSteps=time_horizon
    ))
    simulation.reporters.append(
        StateDataReporter(
            csv_file_name,
            reportInterval=args.freq_csv,
            time=True,
            potentialEnergy=True,
            totalEnergy=True,
            temperature=True,
    ))
    
    # Start simulation!!
    print(">> Starting simulation...")
    simulation.step(time_horizon)
    print(">> Simulation finished!!!")
    
    simulation.minimizeEnergy()
    