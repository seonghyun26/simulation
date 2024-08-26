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
        self.molecule = config['molecule']
        self.state = config['state']
        self.temperature = config['temperature']
        self.time = config['time']
        self.force_field = config['force_field']
        self.solvent = config['solvent']
        self.platform = config['platform']
        self.precision = config['precision']
        
        data_x_list = []
        data_y_list = []
        for t in tqdm(
            range(self.time -1),
            desc="Loading data"
        ):
            current_state = torch.tensor(loaded_traj[t].xyz.squeeze())
            next_state = torch.tensor(loaded_traj[t+1].xyz.squeeze())
            data_x_list.append(current_state)
            data_y_list.append(next_state)
        self.x = torch.stack(data_x_list)
        self.y = torch.stack(data_y_list)
        
        self.sanity_check(loaded_traj)
    
    def sanity_check(self, loaded_traj):
        # print("Running sanity check...")
        # print(f">> x size: {self.x.shape}")
        # print(f">> y size: {self.y.shape}")
        assert torch.equal(x.shape, y.shape), f"Shape of x and y not equal"
        
        for t in tqdm(
            range(self.time -1),
            desc="Sanity check"
        ):
            x = self.x[t]
            y = self.y[t]
            x_frame = torch.tensor(loaded_traj[t].xyz.squeeze())
            y_frame = torch.tensor(loaded_traj[t+1].xyz.squeeze())
            
            assert torch.equal(x, x_frame), f"Frame {t}, x not equal"
            assert torch.equal(y, y_frame), f"Frame {t+1}, y not equal"        
            

    def __getitem__(self, index):
	    return self.x[index], self.y[index]
 
    def __len__(self):
	    return self.x.shape[0]


if __name__ == "__main__":
    # Load config
    # for key, value in vars(args).items():
    #     print(f">> {key}: {value}")
    result_dir = f"../log/{args.molecule}/{args.temperature}/{args.state}"
    pdb_file = f"../data/{args.molecule}/{args.state}.pdb"
    arg_file = f"{result_dir}/args.json"
    with open(arg_file, 'r') as f:
        config = json.load(f)
        # print(">> Loaded config")
        # pprint.pprint(config)
    
    # Load trajectory
    # print("Loading trajectory...")
    start = time.time()
    loaded_traj = mdtraj.load(
        f"{result_dir}/traj.dcd",
        top=pdb_file
    )
    end = time.time()
    # print(f"Done.!! ({end - start:.2f} sec)\n")
    
    # Build dataset
    print(f"Building dataset for {args.temperature}K, sttate {args.molecule} ...")
    dataset = MD_Dataset(loaded_traj, config)
    save_dir = f"../dataset/{args.molecule}/{args.temperature}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        raise ValueError(f"Directory {save_dir} already exists")
    torch.save(dataset, f"{save_dir}/{args.state}.pt")
    print("Done.!!")