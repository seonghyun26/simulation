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
            random_indices = np.random.choice(self.time - 1, args.data_size, replace=True)
            for t in tqdm(
                random_indices,
                desc="Loading data by random idx"
            ):
                current_state = torch.tensor(loaded_traj[t].xyz.squeeze()).to(self.device)
                next_state = torch.tensor(loaded_traj[t+1].xyz.squeeze()).to(self.device)
                for i in range(4):
                    random_interval = random.sample(range(1, np.min([self.time - t, args.max_path_length])), 1)[0]
                    goal_state = torch.tensor(loaded_traj[t+random_interval].xyz.squeeze()).to(self.device)
                    
                    data_x_list.append(current_state)
                    data_y_list.append(next_state)
                    data_goal_list.append(goal_state)
                    data_interval_list.append(torch.tensor(random_interval).to(self.device).unsqueeze(0))
        elif args.index == "goal":
            random_indices = np.random.choice(self.time - 1, args.data_size, replace=True)
            for t in tqdm(
                random_indices,
                desc="Loading data by random idx"
            ):
                random_sim = random.sample(range(sim_num), 1)[0]
                current_state = torch.tensor(loaded_traj[t+1].xyz.squeeze()).to(self.device)
                next_state = torch.tensor(loaded_traj[t].xyz.squeeze()).to(self.device)
                goal_state = torch.tensor(loaded_traj[0].xyz.squeeze()).to(self.device)
                
                data_x_list.append(current_state)
                data_y_list.append(next_state)
                data_goal_list.append(goal_state)
                data_interval_list.append(torch.tensor(t+1).to(self.device).unsqueeze(0))
        elif args.index == "several":
            sim_num = len(loaded_traj)
            random_indices = np.random.choice(self.time - 1, args.data_size, replace=True)
            for t in tqdm(
                random_indices,
                desc="Loading data by random idx"
            ):
                random_sim = random.sample(range(sim_num), 1)[0]
                current_state = torch.tensor(loaded_traj[random_sim][t].xyz.squeeze()).to(self.device)
                next_state = torch.tensor(loaded_traj[random_sim][t+1].xyz.squeeze()).to(self.device)
                goal_state = torch.tensor(loaded_traj[random_sim][0].xyz.squeeze()).to(self.device)
                
                data_x_list.append(current_state)
                data_y_list.append(next_state)
                data_goal_list.append(goal_state)
                data_interval_list.append(torch.tensor(t+1).to(self.device).unsqueeze(0))
        elif args.index == "multi-next":
            random_indices = np.random.choice(self.time - 1, args.data_size, replace=True)
            self.set_simulation()
            for t in tqdm(
                random_indices,
                desc="Multi goal dataset construction"
            ):
                current_state = torch.tensor(loaded_traj[t].xyz.squeeze()).to(self.device)
                
                # Short simulation, get next_state and goal_state
                for i in range(8):
                    random_interval = random.randint(2, 100)
                    next_state, goal_state = self.short_simulation(current_state, random_interval)
                    data_x_list.append(current_state)
                    data_y_list.append(next_state)
                    data_goal_list.append(goal_state)
                    data_interval_list.append(torch.tensor(random_interval).to(self.device).unsqueeze(0))
            self.simulation = None
        elif args.index == "two-step":
            random_indices = np.random.choice(self.time - 2, args.data_size, replace=True)
            for t in tqdm(
                random_indices,
                desc="Two step dataset construction"
            ):
                current_state = torch.tensor(loaded_traj[t].xyz.squeeze()).to(self.device)
                next_state = torch.tensor(loaded_traj[t+1].xyz.squeeze()).to(self.device)
                goal_state = torch.tensor(loaded_traj[t+2].xyz.squeeze()).to(self.device)

                data_x_list.append(current_state)
                data_y_list.append(next_state)
                data_goal_list.append(goal_state)
                data_interval_list.append(torch.tensor(2).to(self.device).unsqueeze(0))
        else:
            for t in tqdm(
                range(args.data_size),
                desc=f"Creating {args.data_size} size dataset"
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
    
    def set_simulation(self):
        pdb_file = PDBFile(f"../data/{self.molecule}-300.0/{self.state}.pdb")
        force_field = ForceField("amber99sbildn.xml", "tip3p.xml")
        system = force_field.createSystem(
            pdb_file.topology,
            nonbondedCutoff=3 * nanometer,
            constraints=HBonds
        )
        integrator = LangevinIntegrator(
            self.temperature * kelvin,
            1 / picosecond,
            1 * femtoseconds
        )
        platform = Platform.getPlatformByName("OpenCL")
        properties = {'Precision': "mixed"}
        simulation = Simulation(
            pdb_file.topology,
            system,
            integrator,
            platform,
            properties
        )
        self.simulation = simulation
        
    def short_simulation(self, current_state, step=100):
        atom_xyz = current_state.detach().cpu().numpy()
        atom_list = [Vec3(atom[0], atom[1], atom[2]) for atom in atom_xyz]
        current_state_openmm = Quantity(value=atom_list, unit=nanometer)
        self.simulation.context.setPositions(current_state_openmm)
        self.simulation.context.setVelocities(Quantity(value=np.zeros((len(atom_list), 3)), unit=nanometer/picosecond))
        self.simulation.minimizeEnergy()
        
        self.simulation.step(1)
        next_state = torch.tensor(self.simulation.context.getState(getPositions=True).getPositions().value_in_unit(nanometer)).to(self.device)
        self.simulation.step(step-1)
        goal_state = torch.tensor(self.simulation.context.getState(getPositions=True).getPositions().value_in_unit(nanometer)).to(self.device)
        
        return next_state, goal_state


# Simluation arguments
parser = argparse.ArgumentParser(description="Dataset builder")

parser.add_argument("--molecule", type=str, help="Path to the PDB file", default="alanine")
parser.add_argument("--state", type=str, help="Molecule state to start the simulation", default="c5")
parser.add_argument("--temperature", type=float, help="Temperature to use", default=273.0)
parser.add_argument("--index", type=str, help="Indexing at dataset", default="")
parser.add_argument("--data_size", type=int, help="Dataset size", default=1000)
parser.add_argument("--sim_len", type=int, help="Length of simulation", default=1000000)
parser.add_argument("--sim_num", type=int, help="Number of simulations", default=1)
parser.add_argument("--dataset_index", type=str, help="Dataset index", default="v1")
parser.add_argument("--max_path_length", type=int, help="Max path length to goal state", default=1000)

args = parser.parse_args()


if __name__ == "__main__":
    # Load config
    result_dir = f"../log/{args.molecule}/{args.temperature}/{args.state}"
    pdb_file = f"../data/{args.molecule}-300.0/{args.state}.pdb"
    arg_file = f"{result_dir}/args.json"
    with open(arg_file, 'r') as f:
        config = json.load(f)
        print(">> Loading config")
        pprint.pprint(config)
        
    
    # Check directory
    save_dir = f"../dataset/{args.molecule}/{args.temperature}"
    file_name = f"{args.state}-{args.data_size}-{args.index}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if os.path.exists(f"{save_dir}/{file_name}.pt"):
        raise ValueError(f"Folder {save_dir}/{file_name}.pt already exists")
    
    
    # Load trajectory
    print(">> Loading trajectory...")
    # start = time.time()
    traj_list = []
    # for i in range(0, args.sim_num):
    loaded_traj = mdtraj.load(
        f"{result_dir}/traj.dcd",
        top=pdb_file
    )
    # traj_list.append(loaded_traj)
    # end = time.time()
    
    # Build dataset
    torch.save(
        MD_Dataset(loaded_traj, config, args, sanity_check=True),
        f"{save_dir}/{file_name}-{args.dataset_index}.pt"
    )
    print(f"Dataset created")
    
