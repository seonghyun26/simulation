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


ALANINE_HEAVY_ATOM_IDX = [
    1, 4, 5, 6, 8, 10, 14, 15, 16, 18
]
POSITIVE_SAMPLE_AUGMENTATION = 0
NEGATIVE_SAMPLE_AUGMENTATION = 100000

class CL_dataset(Dataset):
    def __init__(
        self,
        data_list,
        data_positive_list,
        data_negative_list,
        temperature_list,
        energy_list
    ):
        super(CL_dataset, self).__init__()
        self.device = "cpu"
        
        self.x = data_list.to(self.device)
        self.x_augmented = data_positive_list.to(self.device)
        self.x_augmented_hard = data_negative_list.to(self.device)
        self.temperature = temperature_list.to(self.device)
        self.energy = energy_list.to(self.device)
        
    def __getitem__(self, index):
	    return self.x[index], self.x_augmented[index], self.x_augmented_hard[index], self.temperature[index], self.energy[index]
 
    def __len__(self):
	    return self.x.shape[0]
 


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
    cfg_list,
    energy_list,
    preprocess = "coordinate"
):
    dataset_size = args.dataset_size
    number_of_traj = len(traj_list)
    xyz_list = []
    current_energy_list = []
    xyz_positive_list = []
    xyz_negative_list = []
    distance_list = []
    distance_positive_list = []
    distance_negative_list = []
    temperature_list = []
    
    random_idx_list = []
    for i in range(number_of_traj):
        time_horizon = cfg_list[i]["time"]
        random_idx_list.append(np.random.choice(time_horizon - 2 - NEGATIVE_SAMPLE_AUGMENTATION, dataset_size, replace=True))
    
    for i in tqdm(
        range(dataset_size),
        desc = "Sampling frames from trajectories"
    ):
        '''
            - positive sample: next frame
            - negative sample: future frame
        '''
        for j in range(number_of_traj):
            frame_idx = random_idx_list[j][i]
            current_frame = torch.tensor(traj_list[j][frame_idx].xyz.squeeze())
            xyz_list.append(current_frame)
            distance_list.append(coordinate2distance(current_frame))
            current_energy_list.append(energy_list[j][frame_idx])
            
            next_frame = torch.tensor(traj_list[j][frame_idx + POSITIVE_SAMPLE_AUGMENTATION].xyz.squeeze())
            xyz_positive_list.append(next_frame)
            distance_positive_list.append(coordinate2distance(next_frame))
            
            # augment_idx = np.min([time_horizon - frame_idx - 1, np.random.randint(1, NEGATIVE_SAMPLE_AUGMENTATION)])
            future_frame = torch.tensor(traj_list[j][frame_idx + NEGATIVE_SAMPLE_AUGMENTATION].xyz.squeeze())
            xyz_negative_list.append(future_frame)
            distance_negative_list.append(coordinate2distance(future_frame))
            
            temperature_list.append(torch.tensor(cfg_list[j]["temperature"]))
        '''
            - positive sample: next frame
            - negative sample: same idx frame in another trajectory
        '''
        for j in range(number_of_traj):
            frame_idx = random_idx_list[j][i]
            current_frame = torch.tensor(traj_list[j][frame_idx].xyz.squeeze())
            xyz_list.append(current_frame)
            distance_list.append(coordinate2distance(current_frame))
            current_energy_list.append(energy_list[j][frame_idx])
            
            next_frame = torch.tensor(traj_list[j][frame_idx + POSITIVE_SAMPLE_AUGMENTATION].xyz.squeeze())
            xyz_positive_list.append(next_frame)
            distance_positive_list.append(coordinate2distance(next_frame))
            
            # augment_idx = np.min([time_horizon - frame_idx - 1, np.random.randint(1, 20)])
            other_trajectory_idx = 1 if j == 0 else 0
            future_frame = torch.tensor(traj_list[other_trajectory_idx][frame_idx].xyz.squeeze())
            xyz_negative_list.append(future_frame)
            distance_negative_list.append(coordinate2distance(future_frame))
            
            temperature_list.append(torch.tensor(cfg_list[j]["temperature"]))
    
    xyz_list = torch.stack(xyz_list)
    xyz_positive_list = torch.stack(xyz_positive_list)
    xyz_negative_list = torch.stack(xyz_negative_list)
    distance_list = torch.stack(distance_list)
    distance_positive_list = torch.stack(distance_positive_list)
    distance_negative_list = torch.stack(distance_negative_list)
    temperature_list = torch.stack(temperature_list)
    current_energy_list = torch.stack(current_energy_list)
    print(current_energy_list)
    
    return (xyz_list, xyz_positive_list, xyz_negative_list, temperature_list, current_energy_list), \
        (distance_list, distance_positive_list, distance_negative_list, temperature_list, current_energy_list)

args = init_cl_dataset_args()

if __name__ == "__main__":
    traj_list = []
    energy_list = []
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
        
        # Read trajectory file
        pdb_file = f"./data/{args.molecule}-stable/{state}.pdb"
        loaded_traj = mdtraj.load(
            f"{dir}/traj.dcd",
            top=pdb_file
        )
        traj_list.append(loaded_traj)
        
        # Read energy file
        scalars = pandas.read_csv(f"{dir}/scalars.csv", usecols=[1])
        energy_list.append(torch.tensor(scalars.values.squeeze()))
    
    # Check dataset directory
    save_dir = f"../data/dataset/{args.molecule}/{args.temperature}/{args.dataset_version}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if os.path.exists(f"{save_dir}/cl-xyz.pt"):
        print(f"Dataset-xyz already exists at {save_dir}")
    if os.path.exists(f"{save_dir}/cl-distance.pt"):
        print(f"Dataset-distance already exists at {save_dir}")
    
    # Create CL dataset
    print("\n>> Building CL Dataset...")
    xyz_data, distance_data = traj2dataset(
        traj_list,
        cfg_list,
        energy_list,
        preprocess = "both"
    )
    if not os.path.exists(f"{save_dir}/cl-xyz.pt"):
        torch.save(
            CL_dataset(
                xyz_data[0],
                xyz_data[1],
                xyz_data[2],
                xyz_data[3],
                xyz_data[4]
            ),
            f"{save_dir}/cl-xyz.pt"
        )
    if not os.path.exists(f"{save_dir}/cl-distance.pt"):
        torch.save(
            CL_dataset(
                distance_data[0],
                distance_data[1],
                distance_data[2],
                distance_data[3],
                distance_data[4],
            ),
            f"{save_dir}/cl-distance.pt"
        )
    with open(f"{save_dir}/config.json", 'w') as f:
        json.dump(cfg_list, f, indent=4)
    print(f"Dataset created at {save_dir}")