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
parser.add_argument("--temperature", type=float, help="Temperature to use", default=273.0)
parser.add_argument("--index", type=str, help="Indexing at dataset", default="")
parser.add_argument("--percent", type=int, help="How much precent to use from dataset", default=10)

args = parser.parse_args()


class MD_Dataset(Dataset):
    def __init__(self, loaded_traj, config, args):
        self.molecule = config['molecule']
        self.state = config['state']
        self.temperature = config['temperature']
        self.time = config['time']
        self.force_field = config['force_field']
        self.solvent = config['solvent']
        self.platform = config['platform']
        self.precision = config['precision']
        self.device = "cuda"
        
        data_x_list = []
        data_y_list = []
        data_interval_list = []
        traj = loaded_traj.xyz.squeeze()
        
        if args.index == "random":
            random_indices = random.sample(range(0, self.time - 1), self.time // args.percent)
            for t in tqdm(
                random_indices,
                desc="Loading data by random idx"
            ):
                random_interval = random.sample(range(1, self.time - t), 1)[0]
                current_state = torch.tensor(loaded_traj[t].xyz.squeeze()).to(self.device)
                next_state = torch.tensor(loaded_traj[t+random_interval].xyz.squeeze()).to(self.device)
                data_x_list.append(current_state)
                data_y_list.append(next_state)
                data_interval_list.append(random_interval)
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
        self.delta_t = torch.tensor(data_interval_list).to(self.device)
        
        # self.sanity_check(loaded_traj)
    
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
            x_frame = torch.tensor(loaded_traj[t].xyz.squeeze()).to(self.device)
            y_frame = torch.tensor(loaded_traj[t+1].xyz.squeeze()).to(self.device)
            
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
    dataset = MD_Dataset(loaded_traj, config, args)
    save_dir = f"../dataset/{args.molecule}/{args.temperature}"
    file_name = f"{args.state}-{args.index}"
    print(f"Creating dataset for {args.temperature}K, state {args.state} at {save_dir}/{file_name}...")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    if os.path.exists(f"{save_dir}/{file_name}.pt"):
        raise ValueError(f"Folder {save_dir}/{file_name}.pt already exists")
    else:
        torch.save(dataset, f"{save_dir}/{file_name}.pt")
    
    print(f"Dataset {save_dir}/{file_name} created!!!")
