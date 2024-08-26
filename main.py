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
    
    # Prepare simluation
    start_pdb = set_molecule(args.molecule, args.state)
    forcefield_files = set_force_field(args.force_field, args.solvent)
    platform, properties = set_platform(args.platform, args.precision)
    simulation = set_simulation(args, forcefield_files, start_pdb, platform, properties)
    simulation.reporters = set_logging(args)
    
    print(f">> Molecule: {args.molecule}")
    print(f">> Force field: {forcefield_files}")
    print(f">> Time horizon : {args.time}")
    print(f">> Temperature: {args.temperature}")
    print(f">> Platform, precision: {args.platform}, {args.precision}")
    
    
    # Start simulation!!
    print(">> Starting simulation...")
    simulation.step(time_horizon)
    print(">> Simulation finished!!!")

    