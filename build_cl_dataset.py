import json
import torch
import pprint
import mdtraj
import random
import pandas

import numpy as np

from tqdm import tqdm

from openmm import *
from openmm.app import *
from openmm.unit import *

from torch.utils.data import Dataset

from util.ic import load_ic_transform
from util.dataset_config import init_cl_dataset_args


class CL_dataset(Dataset):
    def __init__(
        self,
        data_list,
        data_augmented_list,
        data_augmented_hard_list,
        temperature_list,
    ):
        super(CL_dataset, self).__init__()
        self.device = "cpu"
        
        self.x = data_list.to(self.device)
        self.x_augmented = data_augmented_list.to(self.device)
        self.x_augmented_hard = data_augmented_hard_list.to(self.device)
        self.temperature = temperature_list.to(self.device)
        
    def __getitem__(self, index):
	    return self.x[index], self.x_augmented[index], self.x_augmented_hard, self.temperature[index]
 
    def __len__(self):
	    return self.x.shape[0]
 

ALANINE_HEAVY_ATOM_IDX = [
    1, 4, 5, 6, 8, 10, 14, 15, 16, 18
]

def coordinate2distance(position):
    position = position.reshape(-1, 3)
    heavy_atom_position = position[ALANINE_HEAVY_ATOM_IDX]
    num_heavy_atoms = len(heavy_atom_position)
    distance = []
    for i in range(num_heavy_atoms):
        for j in range(i+1, num_heavy_atoms):
            distance.append(torch.norm(heavy_atom_position[i] - heavy_atom_position[j]))
    distance = torch.stack(distance)
    
    return distance

def traj2dataset(
    traj_list,
    cfg_list
):
    dataset_size = args.dataset_size
    number_of_traj = len(traj_list)
    data_list = []
    data_augmented_list = []
    data_augmented_hard_list = []
    temperature_list = []
    
    def preprocess_frame(frame):
        if args.preprocess == "coordinate":
            return frame
        elif args.preprocess == "distance":
            return coordinate2distance(frame)
        else:
            raise ValueError(f"Preprocess {args.preprocess} not found")
    
    random_idx_list = []
    for i in range(number_of_traj):
        time_horizon = cfg_list[i]["time"]
        random_idx_list.append(np.random.choice(time_horizon - 2, dataset_size, replace=True))
    
    for i in tqdm(
        range(dataset_size),
        desc = "Sampling frames from trajectories"
    ):
        for j in range(number_of_traj):
            frame_idx = random_idx_list[j][i]
            augment_idx = np.min([time_horizon - frame_idx - 1, np.random.randint(1, 10)])
            current_frame = torch.tensor(traj_list[j][frame_idx].xyz.squeeze())
            data_list.append(preprocess_frame(current_frame))
            next_frame = torch.tensor(traj_list[j][frame_idx+1].xyz.squeeze())
            data_augmented_list.append(preprocess_frame(next_frame))
            future_frame = torch.tensor(traj_list[j][frame_idx + augment_idx].xyz.squeeze())
            data_augmented_hard_list.append(preprocess_frame(future_frame))
            temperature_list.append(torch.tensor(cfg_list[j]["temperature"]))
    
    data_list = torch.stack(data_list)
    data_augmented_list = torch.stack(data_augmented_list)
    data_augmented_hard_list = torch.stack(data_augmented_hard_list)
    temperature_list = torch.stack(temperature_list)
    
    return data_list, data_augmented_list, data_augmented_hard_list, temperature_list

args = init_cl_dataset_args()

if __name__ == "__main__":
    traj_list = []
    cfg_list = []
    simulation_dir = f"./log/{args.molecule}/{args.temperature}"
    
    # Load trajectories
    for traj_dir in tqdm(
        args.traj_dir,
        desc = "Loading trajecatory files"
    ):
        dir = f"{simulation_dir}/{traj_dir}"
        with open(f"{dir}/args.json", 'r') as f:
            config = json.load(f)
            cfg_list.append(config)
            state = config["state"]
        
        pdb_file = f"./data/{args.molecule}-stable/{state}.pdb"
        loaded_traj = mdtraj.load(
            f"{dir}/traj.dcd",
            top=pdb_file
        )
        traj_list.append(loaded_traj)
    
    # Check dataset directory
    save_dir = f"./dataset-CL/{args.molecule}/{args.temperature}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if os.path.exists(f"{save_dir}/{args.dataset_version}.pt"):
        raise ValueError(f"Folder {save_dir}/{args.dataset_version}.pt already exists")
    
    # Create CL dataset
    print("\n>> Building CL Dataset...")
    data_list, data_augmented_list, data_augmented_hard_list, temperature_list = traj2dataset(
        traj_list,
        cfg_list
    )
    torch.save(
        CL_dataset(
            data_list,
            data_augmented_list,
            data_augmented_hard_list,
            temperature_list
        ),
        f"{save_dir}/{args.dataset_version}.pt"
    )
    print(f"Dataset created at {save_dir}/{args.dataset_version}.pt")
    
