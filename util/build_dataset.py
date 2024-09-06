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


# MD dataset class
class MD_Dataset(Dataset):
    def __init__(
        self,
        loaded_traj,
        config,
        args,
        sanity_check=False
    ):
        super(MD_Dataset, self).__init__()
        
        self.molecule = config['molecule']
        self.state = config['state']
        self.temperature = config['temperature']
        self.time = config['time']
        self.force_field = config['force_field']
        self.solvent = config['solvent']
        self.platform = config['platform']
        self.precision = config['precision']
        self.device = "cpu"
        
        data_x_list = []
        data_y_list = []
        data_interval_list = []
        data_goal_list = []
        
        if args.index == "random":
            random_indices = np.random.choice(self.time - 1, self.time // args.percent, replace=True)
            for t in tqdm(
                random_indices,
                desc="Loading data by random idx"
            ):
                current_state = torch.tensor(loaded_traj[t].xyz.squeeze()).to(self.device)
                next_state = torch.tensor(loaded_traj[t+1].xyz.squeeze()).to(self.device)
                random_interval = random.sample(range(1, self.time - t), 1)[0]
                goal_state = torch.tensor(loaded_traj[t+random_interval].xyz.squeeze()).to(self.device)
                
                data_x_list.append(current_state)
                data_y_list.append(next_state)
                data_goal_list.append(goal_state)
                data_interval_list.append(torch.tensor(random_interval).to(self.device).unsqueeze(0))
        else:
            for t in tqdm(
                range((self.time -1) // args.percent),
                desc=f"Loading {args.precent} precent of dataset from initial frame"
            ):
                current_state = torch.tensor(loaded_traj[t].xyz.squeeze()).to(self.device)
                next_state = torch.tensor(loaded_traj[t+1].xyz.squeeze()).to(self.device)
                data_x_list.append(current_state)
                data_y_list.append(next_state)
                data_interval_list.append(1)
                
        self.x = torch.stack(data_x_list).to(self.device)
        self.y = torch.stack(data_y_list).to(self.device)
        self.goal = torch.stack(data_goal_list).to(self.device)
        self.delta_time = torch.stack(data_interval_list).to(self.device)
        
        # if sanity_check:
        #     self.sanity_check(loaded_traj)
        
    def sanity_check(self, loaded_traj):
        assert torch.equal(self.x.shape, self.y.shape), f"Shape of x and y not equal"
        
        for t in tqdm(
            range(self.time -1),
            desc="Sanity check"
        ):
            x = self.x[t]
            y = self.y[t]
            x_frame = torch.tensor(loaded_traj[t].xyz.squeeze()).to(self.device)
            y_frame = torch.tensor(loaded_traj[t+1].xyz.squeeze()).to(self.device)
            
            assert torch.equal(x, x_frame), f"Frame {t}, x not equal"
            assert torch.equal(y, y_frame), f"Frame {t+1}, y not equal"
        
    def __getitem__(self, index):
	    return self.x[index], self.y[index], self.goal[index], self.delta_time[index]
 
    def __len__(self):
	    return self.x.shape[0]


# Simluation arguments
parser = argparse.ArgumentParser(description="Dataset builder")

parser.add_argument("--molecule", type=str, help="Path to the PDB file", default="alanine")
parser.add_argument("--state", type=str, help="Molecule state to start the simulation", default="c5")
parser.add_argument("--temperature", type=float, help="Temperature to use", default=273.0)
parser.add_argument("--index", type=str, help="Indexing at dataset", default="")
parser.add_argument("--percent", type=int, help="How much precent to use from dataset", default=10)
parser.add_argument("--verbose", type=bool, help="Verbose mode", default=True)

args = parser.parse_args()

def print_verbose(msg, verbose):
    if verbose:
        pprint.pprint(msg)

if __name__ == "__main__":
    # Load config
    for key, value in vars(args).items():
        print_verbose(f">> {key}: {value}", args.verbose)
    result_dir = f"../log/{args.molecule}/{args.temperature}/{args.state}"
    pdb_file = f"../data/{args.molecule}/{args.state}.pdb"
    arg_file = f"{result_dir}/args.json"
    with open(arg_file, 'r') as f:
        config = json.load(f)
        print_verbose(">> Loaded config", args.verbose)
        print_verbose(config, args.verbose)
        
    
    # Check directory
    save_dir = f"../dataset/{args.molecule}/{args.temperature}"
    file_name = f"{args.state}-{args.index}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if os.path.exists(f"{save_dir}/{file_name}.pt"):
        raise ValueError(f"Folder {save_dir}/{file_name}.pt already exists")
    
    
    # Load trajectory
    print_verbose("Loading trajectory...", args.verbose)
    start = time.time()
    loaded_traj = mdtraj.load(
        f"{result_dir}/traj.dcd",
        top=pdb_file
    )
    end = time.time()
    print_verbose(f"Done.!! ({end - start:.2f} sec)\n", args.verbose)
    
    
    # Build dataset
    torch.save(
        MD_Dataset(loaded_traj, config, args, sanity_check=True),
        f"{save_dir}/{file_name}.pt"
    )
    print(f"Dataset {save_dir}/{file_name} created!!!")
