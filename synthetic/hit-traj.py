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
parser.add_argument("--num_steps", default=1000, type=int)
parser.add_argument("--timestep", default=0.01, type=float)
parser.add_argument("--num_samples", default=1024, type=int)
parser.add_argument("--temperature", default=1200, type=float)

args = parser.parse_args()


class Synthetic:
    def __init__(self, args):
        self.kB = 8.6173303e-5
        self.std = np.sqrt(2 * self.kB * args.temperature * args.timestep)
        self.log_prob = torch.distributions.Normal(0, self.std).log_prob
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
    for name in ["hit", "traj"]:
        if not os.path.exists(f"{args.save_dir}/{name}"):
            os.makedirs(f"{args.save_dir}/{name}")

    cnt = 0
    mds = Synthetic(args)

    positions = torch.zeros(
        (10000, args.num_steps + 1, 2),
        device=args.device,
    )

    while cnt < args.num_samples:
        noises = torch.normal(
            torch.zeros(
                (10000, args.num_steps, 2),
                device=args.device,
            ),
            torch.ones(
                (10000, args.num_steps, 2),
                device=args.device,
            ),
        )
        position = mds.start_position.unsqueeze(0)
        force = mds.energy_function(position)[0]
        positions[:, 0] = position
        for s in tqdm(range(args.num_steps), desc="Sampling"):
            position = position + force * args.timestep + mds.std * noises[:, s]
            force = mds.energy_function(position)[0]
            positions[:, s + 1] = position

        
        # Save hit
        hit = (positions[:, -1] - mds.target_position.unsqueeze(0)).square().sum(
            -1
        ).sqrt() < 0.2
        for i in tqdm(range(10000), desc="Saving hits"):
            if hit[i]:
                np.save(
                    f"{args.save_dir}/traj/{cnt}.npy", positions[i].cpu().numpy()
                )
                np.save(
                    f"{args.save_dir}/hit/{cnt}.npy", positions[i].cpu().numpy()
                )
                cnt += 1
                print(f"Sampled {cnt}/{args.num_samples}")
                if cnt == args.num_samples:
                    break
                
        print(f"Saved {cnt} samples.")