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

from util.dataset_config import init_dataset_args

# MD dataset class
class MD_Dataset(Dataset):
    def __init__(
        self,
        traj_list,
        config,
        args,
        sanity_check=False
    ):
        super(MD_Dataset, self).__init__()
        
        self.molecule = config['molecule']
        self.state = config['state']
        self.time = config['time']
        self.force_field = config['force_field']
        self.solvent = config['solvent']
        self.platform = config['platform']
        self.precision = config['precision']
        self.args = args
        
        data_x_list = []
        data_y_list = []
        data_interval_list = []
        data_goal_list = []
        data_temp_list = []
        
        random_indices = np.random.choice(self.time - 2, args.sim_length, replace=True)
        
        if args.dataset_type == "multi-temp":
            temp_list = [200, 400, 600]
            # self.set_simulation()
            for t in tqdm(
                random_indices,
                desc="Multi temperature dataset construction"
            ):
                for i in range(3):
                    loaded_traj = traj_list[i]
                    current_state = torch.tensor(loaded_traj[t].xyz.squeeze())
                    random_interval = random.sample(range(1, np.min([self.time - t, args.max_path_length])), 1)[0]
                    next_state = torch.tensor(loaded_traj[t+1].xyz.squeeze())
                    goal_state = torch.tensor(loaded_traj[t+random_interval].xyz.squeeze())
                    
                    data_x_list.append(current_state)
                    data_y_list.append(next_state)
                    data_goal_list.append(goal_state)
                    data_interval_list.append(torch.tensor(random_interval).unsqueeze(0))
                    data_temp_list.append(torch.tensor(temp_list[i]).unsqueeze(0))
            # self.simulation = None
        else:
            raise ValueError(f"Index {args.dataset_type} not found")
                
        self.x = torch.stack(data_x_list)
        self.y = torch.stack(data_y_list)
        self.goal = torch.stack(data_goal_list)
        self.delta_time = torch.stack(data_interval_list)
        self.temperature = torch.stack(data_temp_list)
        
        
    def sanity_check(self, loaded_traj):
        assert torch.equal(self.x.shape, self.y.shape), f"Shape of x and y not equal"
        
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
	    return self.x[index], self.y[index], self.goal[index], self.delta_time[index], self.temperature[index]
 
    def __len__(self):
	    return self.x.shape[0]
    


args = init_dataset_args()
if __name__ == "__main__":
    # Load config
    result_dir = f"./log/{args.molecule}/{args.temperature}/{args.state}"
    pdb_file = f"./data/{args.molecule}-stable/{args.state}.pdb"
    arg_file = f"{result_dir}/args.json"
    with open(arg_file, 'r') as f:
        config = json.load(f)
        print(">> Loading config")
        pprint.pprint(config)
        
    
    # Check directory
    save_dir = f"./dataset/{args.molecule}/{args.temperature}"
    file_name = f"{args.state}-{args.dataset_type}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if os.path.exists(f"{save_dir}/{file_name}.pt"):
        raise ValueError(f"Folder {save_dir}/{file_name}.pt already exists")
    
    
    # Load trajectory
    print("\n>> Loading trajectory...")
    traj_list = []
    temp_list = ["200.0", "400.0", "600.0"]
    for temp in tqdm(temp_list):
        result_dir = f"./log/{args.molecule}/{temp}/{args.state}"
        loaded_traj = mdtraj.load(
            f"{result_dir}/traj.dcd",
            top=pdb_file
        )
        traj_list.append(loaded_traj)
    print("Done.")
    
    
    # Build dataset
    print("\n>> Building Dataset...")
    torch.save(
        MD_Dataset(traj_list, config, args, sanity_check=False),
        f"{save_dir}/{file_name}-{args.dataset_version}.pt"
    )
    print(f"Dataset created.")
    
