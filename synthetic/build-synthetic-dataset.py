import json
import torch
import pprint
import mdtraj
import random
import pandas
import argparse

import numpy as np

from tqdm import tqdm

from openmm import *
from openmm.app import *
from openmm.unit import *

from torch.utils.data import Dataset

parser = argparse.ArgumentParser(description="Dataset builder")
parser.add_argument("--molecule", type=str, help="Molecule type", default="double-well")
parser.add_argument("--state", type=str, help="Molecule state to start the simulation", default="left")
parser.add_argument("--temperature", type=str, help="Temperature to use", default="mix")
parser.add_argument("--dataset_size", type=int, help="Dataset size", default=10000)
parser.add_argument("--dataset_type", type=str, help="Dataset building method", default="multi-temp")
parser.add_argument("--dataset_version", type=str, help="Dataset index", default="v1")
parser.add_argument("--max_path_length", type=int, help="Max path length to goal state", default=500)
parser.add_argument("--sim_repeat_num", type=int, help="Number of simulations to do from one state", default=8)


class SyntheticDataset(Dataset):
    def __init__(
        self,
        traj_list,
        config,
        args,
        sanity_check=False
    ):
        super(SyntheticDataset, self).__init__()
        
        self.molecule = args.molecule
        self.state = args.state
        self.time = int(args.dataset_size)
        self.args = args
        if args.temperature == "mix":
            self.temperature = [300.0, 600.0, 900.0, 1200.0]
        else:
            self.temperature = args.temperature
        self.mds = Synthetic()
        
        data_x_list = []
        data_y_list = []
        data_goal_list = []
        data_temp_list = []
        data_interval_list = []
        
        random_indices = np.random.choice(self.time - 2, args.dataset_size, replace=True)
        
        if args.dataset_type == "random":
            for t in tqdm(
                random_indices,
                desc="Random dataset construction"
            ):
                current_state = torch.tensor(loaded_traj[t].xyz.squeeze())
                next_state = torch.tensor(loaded_traj[t+1].xyz.squeeze())
                raise ValueError("Random dataset construction not implemented")
        elif args.dataset_type == "multi-temp":
            for t in tqdm(
                random_indices,
                desc="Multi temperature dataset construction"
            ):
                for temp_idx in range(4):
                    loaded_traj = traj_list[temp_idx].squeeze(0)
                    current_state = torch.tensor(loaded_traj[t])
                    
                    # for i in range(args.sim_repeat_num):
                    random_interval = random.sample(range(1, np.min([self.time - t, args.max_path_length])), 1)[0]
                    next_state = torch.tensor(loaded_traj[t+1])
                    goal_state = torch.tensor(loaded_traj[t+random_interval])
                    
                    data_x_list.append(current_state)
                    data_y_list.append(next_state)
                    data_goal_list.append(goal_state)
                    data_interval_list.append(torch.tensor(random_interval).unsqueeze(0))
                    data_temp_list.append(torch.tensor(float(self.temperature[temp_idx])).unsqueeze(0))
        else:
            raise ValueError(f"Dataset type {args.dataset_type} not found")
                
        self.x = torch.stack(data_x_list)
        self.y = torch.stack(data_y_list)
        self.goal = torch.stack(data_goal_list)
        self.delta_time = torch.stack(data_interval_list)
        self.temperature = torch.stack(data_temp_list)
        
    def __getitem__(self, index):
	    return self.x[index], self.y[index], self.goal[index], self.delta_time[index], self.temperature[index]
 
    def __len__(self):
	    return self.x.shape[0]


class Synthetic:
	def force_energy(self, position):
		position = self.potential(position)
		position.requires_grad_(False)
		force = -torch.autograd.grad(potential.sum(), position)[0]
		return force, potential.detach()

	def potential(self, position):
		position.requires_grad_(True)
		x = position[:, 0]
		y = position[:, 1]
		term_1 = 4 * (1 - x**2 - y**2) ** 2
		term_2 = 2 * (x**2 - 2) ** 2
		term_3 = ((x + y) ** 2 - 1) ** 2
		term_4 = ((x - y) ** 2 - 1) ** 2
		potential = (term_1 + term_2 + term_3 + term_4 - 2.0) / 6.0

		return potential.detach()

args = parser.parse_args()
if __name__ == "__main__":
    # Load config
    result_dir = f"./res/10000-{args.state}"
    config = None
    
    # Check directory
    save_dir = f"../dataset/{args.molecule}/{args.temperature}"
    file_name = f"{args.state}-{args.dataset_type}-{args.dataset_version}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if os.path.exists(f"{save_dir}/{file_name}.pt"):
        raise ValueError(f"Folder {save_dir}/{file_name}.pt already exists")
    
    # Load trajectory
    print("\n>> Loading trajectory...")
    traj_list = []
    for temp in ["300.0", "600.0", "900.0", "1200.0"]:
        loaded_traj = np.load(f"{result_dir}/{temp}/long.npy")
        traj_list.append(loaded_traj)
    print("Done.")
    
    
    # Create dataset
    print("\n>> Building Dataset...")
    torch.save(
        SyntheticDataset(traj_list, config, args, sanity_check=False),
        f"{save_dir}/{file_name}.pt"
    )
    print(f"Dataset created at {save_dir}/{file_name}.pt")
    
