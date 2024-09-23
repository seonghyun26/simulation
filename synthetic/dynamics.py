import os
import torch
import argparse
import numpy as np

from tqdm import tqdm
from datetime import datetime

parser = argparse.ArgumentParser()

# System Config
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--save_dir", default="res", type=str)

# Sampling Config
parser.add_argument("--num_steps", default=100000000, type=int)
parser.add_argument("--timestep", default=0.01, type=float)
parser.add_argument("--temperature", default=1200, type=float)
parser.add_argument("--reverse", default=False, type=bool)

args = parser.parse_args()


class Synthetic:
    def __init__(self, args):
        self.kB = 8.6173303e-5
        self.std = np.sqrt(2 * self.kB * args.temperature * args.timestep)
        self.log_prob = torch.distributions.Normal(0, self.std).log_prob
        if args.reverse:
            self.start_position = torch.tensor([1.118, 0], dtype=torch.float32).to(
                args.device
            )
            self.target_position = torch.tensor([-1.118, 0], dtype=torch.float32).to(
                args.device
            )
        else:
            self.start_position = torch.tensor([-1.118, 0], dtype=torch.float32).to(
                args.device
            )
            self.target_position = torch.tensor([1.118, 0], dtype=torch.float32).to(
                args.device
            )

    def energy_function(self, position):
        position.requires_grad_(True)
        x = position[:, 0]
        y = position[:, 1]
        term_1 = 4 * (1 - x**2 - y**2) ** 2
        term_2 = 2 * (x**2 - 2) ** 2
        term_3 = ((x + y) ** 2 - 1) ** 2
        term_4 = ((x - y) ** 2 - 1) ** 2
        potential = (term_1 + term_2 + term_3 + term_4 - 2.0) / 6.0
        force = -torch.autograd.grad(potential.sum(), position)[0]
        position.requires_grad_(False)
        return force, potential.detach()


if __name__ == "__main__":
    date = datetime.now().strftime("%m%d-%H%M")
    args.save_dir = f"{args.save_dir}/{date}"
    for name in [str(args.temperature)]:
        if not os.path.exists(f"{args.save_dir}/{name}"):
            os.makedirs(f"{args.save_dir}/{name}")

    mds = Synthetic(args)

    positions = torch.zeros(
        (1, args.num_steps + 1, 2),
        device=args.device,
    )

    noises = torch.normal(
        torch.zeros(
            (1, args.num_steps, 2),
            device=args.device,
        ),
        torch.ones(
            (1, args.num_steps, 2),
            device=args.device,
        ),
    )
    start_position = mds.start_position.unsqueeze(0)
    force = mds.energy_function(start_position)[0]
    positions[:, 0] = start_position
    position = start_position
    
    for s in tqdm(range(args.num_steps), desc=f"Running dynamics at T={str(args.temperature)}"):
        position = position + force * args.timestep + mds.std * noises[:, s]
        force = mds.energy_function(position)[0]
        positions[:, s + 1] = position

    np.save(
        f"{args.save_dir}/{str(args.temperature)}/long.npy", positions.cpu().numpy()
    )
            
    print(f"Saved traj.")