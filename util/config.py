import os
import pytz
import json
import argparse

from datetime import datetime

from openmm import *
from openmm.app import *
from openmm.unit import *

def init_args():
    # Parser
    parser = argparse.ArgumentParser(description="Simulation script")

    parser.add_argument("--molecule", type=str, help="Path to the PDB file", default="ad")

    # Simluation physcial environment
    parser.add_argument("--force_field", type=str, help="Force field to use", default="amber14")
    parser.add_argument("--solvent", type=str, help="Solvent to use", default="tip3p")
    parser.add_argument("--temperature", type=float, help="Temperature to use", default=300)
    parser.add_argument("--time", type=int, help="Total simulation steps", default=1e+8)

    # Simluation platform environment
    parser.add_argument("--platform", type=str, help="Platform to use", default="OpenCL")
    parser.add_argument("--precision", type=str, help="Precision to use", default="single")

    # Logging intervals
    parser.add_argument("--freq_dcd", type=int, help="Logging interval for dcd", default="1")
    parser.add_argument("--freq_stdout", type=int, help="Logging interval for stdout", default="10_000")
    parser.add_argument("--freq_csv", type=int, help="Logging interval for csv", default="1_000")

    args = parser.parse_args()

    return args


def set_molecule(molecule_name):
    if molecule_name == "ad-c5":
        pdb_file_name = "alanine/c5.pdb"
    elif molecule_name == "ad-c7ax":
        pdb_file_name = "alanine/c7ax.pdb"
    elif molecule_name == "ad-pii":
        pdb_file_name = "alanine/pII.pdb"
    elif molecule_name == "ad-alpha_L":
        pdb_file_name = "alanine/alpha_L.pdb"
    elif molecule_name == "ad-alpha_R":
        pdb_file_name = "alanine/alpha_R.pdb"
    elif molecule_name == "ad-alpha_P":
        pdb_file_name = "alanine/alpha_P.pdb"
    else:
        raise ValueError(f"Molecule {molecule_name} not recognized")
    
    pdb = PDBFile("./data/" + pdb_file_name) 
    return pdb

def set_force_field(force_field, solvent):
    print(">> Force field: ", force_field)
    print(">> Solvent: ", solvent)
    
    files = []
    
    if force_field == "amber14":
        files.append("amber14-all.xml")
    elif force_field == "amber99":
        files.append("amber99sbildn.xml")
    else:
        raise ValueError(f"Force field {force_field} not recognized")
    
    if solvent == "tip3p":
        # files.append("amber14/tip3pfb.xml")
        files.append("tip3pfb.xml")
    elif solvent == "vacuum":
        pass
    else:
        raise ValueError(f"Solvent {solvent} not recognized")

    return files

def set_platform(platform, precision):
    if platform == "CPU":
        platform = Platform.getPlatformByName('CPU')
        properties = {}
    elif platform == "CUDA":
        # raise ValueError("CUDA does not work")
        platform = Platform.getPlatformByName('CUDA')
        assert precision in ['single', 'mixed', 'double'], f"Precision {precision} not recognized"
        properties = {'DeviceIndex': 0, 'Precision': f"{precision}"}
    elif platform == "OpenCL":
        platform = Platform.getPlatformByName('OpenCL')
        assert precision in ['single', 'mixed', 'double'], f"Precision {precision} not recognized"
        properties = {'Precision': f"{precision}"}
    else:
        raise ValueError(f"Platform {platform} not recognized")
    
    return platform, properties

def set_logging(args):
    kst = pytz.timezone('Asia/Seoul')
    current_date = datetime.now(kst).strftime("%m%d-%H:%M:%S")
    log_dir = f"./log/{args.molecule}/{current_date}"
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    with open(f"{log_dir}/args.json", 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)
        
    return log_dir
    