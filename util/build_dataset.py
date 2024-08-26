import json
import time
import torch
import pprint
import mdtraj
import random
import pandas
import nglview
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from openmm import *
from openmm.app import *
from openmm.unit import *

from torch.utils.data import Dataset

parser = argparse.ArgumentParser(description="Dataset builder")

# Simluation arguments
parser.add_argument("--molecule", type=str, help="Path to the PDB file", default="alanine")
parser.add_argument("--state", type=str, help="Molecule state to start the simulation", default="c5")
parser.add_argument("--temperature", type=float, help="Temperature to use", default=300.0)
args = parser.parse_args()


class MD_Dataset(Dataset):
    def __init__(self, traj, config):
        self.molecule = config.molecule
        self.state = config.state
        self.temperature = config.temperature
        self.time = config.time
        self.force_field = config.force_field
        self.solvent = config.solvent
        self.platform = config.platform
        self.precision = config.precision
        
        for t in tqdm(config['time']):
            current_state = loaded_traj[t].xyz.reshape(-1)
            next_state = loaded_traj[t+1].xyz.reshape(-1)
        

if __name__ == "__main__":
    print("Loading trajectory...")
    for key, value in vars(args).items():
        print(f">> {key}: {value}")
    result_dir = f"../log/{args.molecule}/{args.temperature}/{args.state}"
    pdb_file = f"../data/{args.molecule}/{args.state}.pdb"
    arg_file = f"{result_dir}/args.json"
    with open(arg_file, 'r') as f:
        config = json.load(f)
        print(">> Loaded config")
        pprint.pprint(config)
    start = time.time()
    loaded_traj = mdtraj.load(
        f"{result_dir}/traj.dcd",
        top=pdb_file
    )
    end = time.time()
    print(f"Done.!! ({end - start:.2f} sec)\n")
    
    print("Building dataset...")
    dataset = MD_Dataset(loaded_traj, config)
    torch.save(dataset, f"../dataset/{args.molecule}/{args.temperature}/{args.state}.pt")
    print("Done.!!")