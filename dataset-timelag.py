import os
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

from util.dataset_config import init_cl_dataset_args


ALDP_PHI_ANGLE = [4, 6, 8, 14]
ALDP_PSI_ANGLE = [6, 8, 14, 16]

ALANINE_HEAVY_ATOM_IDX = [
    1, 4, 5, 6, 8, 10, 14, 15, 16, 18
]
CHIGNOLIN_HEAVY_ATOM_IDX = [
    0, 1, 2, 3, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    30, 31, 32, 33, 34, 35, 36, 37, 42, 43, 44, 45, 46, 47, 48,
    56, 57, 58, 59, 60, 61, 62, 63, 64,
    71, 72, 73, 74, 75, 76, 77, 85, 86, 87, 88,
    92, 93, 94, 95, 96, 97, 98, 106, 107, 108, 109,
    110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
    130, 131, 132, 133, 134
]


def check_and_save(
    dir,
    name,
    data
):    
    if not os.path.exists(dir):
        os.makedirs(dir)
    if os.path.exists(f"{dir}/{name}"):
        print(f"{name} already exists at {dir}/{name}")
        pass
    else:
        if isinstance(data, torch.Tensor):
            torch.save(data, f"{dir}/{name}")
        elif isinstance(data, np.ndarray):
            np.save(f"{dir}/{name}", data)
        else:
            raise ValueError(f"Data type {type(data)} not supported")
        print(f"{name} dataset saved at {dir}")

def coordinate2distance(
    position,
    molecule = "alanine"
):
    position = position.reshape(-1, 3)
    if molecule == "alanine":
        heavy_atom_position = position[ALANINE_HEAVY_ATOM_IDX]
    elif molecule == "chignolin":
        heavy_atom_position = position[CHIGNOLIN_HEAVY_ATOM_IDX]
    else:
        raise ValueError(f"Molecule {molecule} not found")
    
    num_heavy_atoms = len(heavy_atom_position)
    distance = []
    for i in range(num_heavy_atoms):
        for j in range(i+1, num_heavy_atoms):
            distance.append(torch.norm(heavy_atom_position[i] - heavy_atom_position[j]))
    distance = torch.stack(distance)
    
    return distance

def kabsch(
    P: torch.Tensor,
    Q: torch.Tensor
) -> torch.Tensor:
    centroid_P = torch.mean(P, dim=-2, keepdims=True)
    centroid_Q = torch.mean(Q, dim=-2, keepdims=True)
    p = P - centroid_P
    q = Q - centroid_Q

    # Compute the covariance matrix
    H = torch.matmul(p.transpose(-2, -1), q)
    U, S, Vt = torch.linalg.svd(H)
    
    d = torch.det(torch.matmul(Vt.transpose(-2, -1), U.transpose(-2, -1)))  # B
    Vt[d < 0.0, -1] *= -1.0

    # Optimal rotation and translation
    R = torch.matmul(Vt.transpose(-2, -1), U.transpose(-2, -1))
    t = centroid_Q - torch.matmul(centroid_P, R.transpose(-2, -1))

    # Calculate RMSD
    P_aligned = torch.matmul(P, R.transpose(-2, -1)) + t
    return P_aligned

def compute_dihedral(
    positions: torch.Tensor
) -> np.ndarray:
    """http://stackoverflow.com/q/20305272/1128289"""
    def dihedral(p):
        if not isinstance(p, np.ndarray):
            p = p.numpy()
        b = p[:-1] - p[1:]
        b[0] *= -1
        v = np.array([v - (v.dot(b[1]) / b[1].dot(b[1])) * b[1] for v in [b[0], b[2]]])
        
        # Normalize vectors
        v /= np.sqrt(np.einsum('...i,...i', v, v)).reshape(-1, 1)
        b1 = b[1] / np.linalg.norm(b[1])
        x = np.dot(v[0], v[1])
        m = np.cross(v[0], b1)
        y = np.dot(m, v[1])
        
        return np.arctan2(y, x)

    # angles = np.array(list(map(dihedral, positions)))
    angle_list = []
    for position in tqdm(
        positions,
        desc = "Computing dihedral angles"
    ):
        angle_list.append(dihedral(position))
    angle_list = np.stack(angle_list)

    return angle_list



def traj2dataset(
    traj_list,
    cfg_list,
):
    molecule = args.molecule
    dataset_size = args.dataset_size
    number_of_traj = len(traj_list)
    current_state_xyz = []
    current_state_distance = []
    time_lagged_state_xyz = []
    time_lagged_state_distance = []
    current_state_phi = []
    current_state_psi = []
    time_lagged_state_phi = []
    time_lagged_state_psi = []
    reference_state_xyz = torch.tensor(traj_list[0][0].xyz.squeeze())
    
    print(f"Sampling {dataset_size} frames from {number_of_traj} trajectories with {traj_list[0].n_frames} frames")
    random_idx = np.random.choice(traj_list[0].n_frames - 2 - args.negative_sample_augmentation, dataset_size, replace=True)
    for traj_idx in range(number_of_traj):
        for i in tqdm(
            range(dataset_size),
            desc = f"Sampling frames from trajectory {traj_idx}"
        ):
            frame_idx = random_idx[i]
            current_frame = torch.tensor(traj_list[traj_idx][frame_idx].xyz.squeeze())
            current_state_xyz.append(kabsch(current_frame, reference_state_xyz))
            current_state_distance.append(coordinate2distance(current_frame))
            time_lagged_frame = torch.tensor(traj_list[traj_idx][frame_idx + args.negative_sample_augmentation].xyz.squeeze())
            time_lagged_state_xyz.append(kabsch(time_lagged_frame, reference_state_xyz))
            time_lagged_state_distance.append(coordinate2distance(time_lagged_frame))
        
    current_state_xyz = torch.stack(current_state_xyz)
    current_state_distance = torch.stack(current_state_distance)
    current_state_phi = compute_dihedral(current_state_xyz[:, ALDP_PHI_ANGLE])
    current_state_psi = compute_dihedral(current_state_xyz[:, ALDP_PSI_ANGLE])
    
    time_lagged_state_xyz = torch.stack(time_lagged_state_xyz)
    time_lagged_state_distance = torch.stack(time_lagged_state_distance)
    time_lagged_state_phi = compute_dihedral(time_lagged_state_xyz[:, ALDP_PHI_ANGLE])
    time_lagged_state_psi = compute_dihedral(time_lagged_state_xyz[:, ALDP_PSI_ANGLE])
    
    return current_state_xyz, current_state_distance, current_state_phi, current_state_psi, \
        time_lagged_state_xyz, time_lagged_state_distance, time_lagged_state_phi, time_lagged_state_psi


args = init_cl_dataset_args()

if __name__ == "__main__":
    traj_list = []
    energy_list = []
    cfg_list = []
    simulation_dir = f"./log/{args.molecule}/{args.temperature}"
    print(f">> Building timelag dataset {args.dataset_version}")
    
    # Load trajectories
    for traj_dir in tqdm(
        args.traj_dir,
        desc = "Loading trajecatory files"
    ):
        # Load configuration file
        dir = f"{simulation_dir}/{traj_dir}"
        with open(f"{dir}/args.json", 'r') as f:
            config = json.load(f)
            cfg_list.append(config)
            state = config["state"]
        
        # Load topology file from pdb
        if args.molecule == "alanine":
            pdb_file = f"./data/{args.molecule}-stable/{state}.pdb"
        elif args.molecule == "chignolin":
            pdb_file = f"./data/{args.molecule}/{state}.pdb"
        else:
            raise ValueError(f"Molecule {args.molecule} not found")
        
        # Load trajectory file
        loaded_traj = mdtraj.load(
            f"{dir}/traj.dcd",
            top=pdb_file
        )
        traj_list.append(loaded_traj)
    
    # Check dataset directory
    save_dir = f"../data/dataset/{args.molecule}/{args.temperature}/{args.dataset_version}"
    for name in ["cl-xyz-aligned.pt", "cl-distance.pt", "phi.npy", "psi.npy", "label.npy"]:
        if os.path.exists(f"{save_dir}/{name}"):
            print(f"{name} already exists at {save_dir}")
            exit()
    
    # Create timelag dataset
    print("\n>> Building timelag Dataset...")
    current_state_xyz, current_state_distance, current_state_phi, current_state_psi, \
    time_lagged_state_xyz, time_lagged_state_distance, time_lagged_state_phi, time_lagged_state_psi = traj2dataset(
        traj_list,
        cfg_list,
    )
    
    check_and_save(dir = save_dir, name = "cl-xyz-aligned.pt", data = current_state_xyz)
    check_and_save(dir = save_dir, name = "cl-distance.pt", data = current_state_distance)
    check_and_save(dir = save_dir, name = "phi.npy", data = current_state_phi)
    check_and_save(dir = save_dir, name = "psi.npy", data = current_state_psi)
    
    check_and_save(dir = save_dir, name = "cl-xyz-aligned-timelag.pt", data = time_lagged_state_xyz)
    check_and_save(dir = save_dir, name = "cl-distance-timelag.pt", data = time_lagged_state_distance)
    check_and_save(dir = save_dir, name = "phi-timelag.npy", data = time_lagged_state_phi)
    check_and_save(dir = save_dir, name = "psi-timelag.npy", data = time_lagged_state_psi)
