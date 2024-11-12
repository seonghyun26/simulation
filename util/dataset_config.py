import argparse

def init_dataset_args():
    parser = argparse.ArgumentParser(description="Dataset builder")

    parser.add_argument("--molecule", type=str, help="Path to the PDB file", default="alanine")
    parser.add_argument("--temperature", type=float, help="Temperature to use", default=600.0)
    parser.add_argument("--state", type=str, help="Molecule state to start the simulation", default="c5")
    parser.add_argument("--dataset_size", type=int, help="Dataset size", default=1000)
    parser.add_argument("--dataset_type", type=str, help="Dataset building method", default="random")
    parser.add_argument("--dataset_version", type=str, help="Dataset version", default="v1")
    parser.add_argument("--max_path_length", type=int, help="Max path length to goal state", default=1000)
    parser.add_argument("--sim_repeat_num", type=int, help="Number of simulations to do from one state", default=4)

    args = parser.parse_args()
    
    return args


def init_cl_dataset_args():
    parser = argparse.ArgumentParser(description="Dataset builder for Contrasstive learning")

    parser.add_argument("--molecule", type=str, help="Path to the PDB file", default="alanine")
    parser.add_argument("--temperature", type=float, help="Temperature to use", default=600.0)
    parser.add_argument('--traj_dir', action='append', help='List of trajectories to make a dataset', required=True)
    parser.add_argument("--dataset_size", type=int, help="Dataset size", default=10000)
    parser.add_argument("--preprocess", type=str, help="Preprocess data input", default="coordinate")
    parser.add_argument("--dataset_version", type=str, help="Dataset version", default="debug-v1")

    args = parser.parse_args()
    
    return args