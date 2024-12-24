import os
import random
import mdtraj
import pandas
import datetime

from rdkit import Chem
from  tqdm import tqdm
from rdkit.Chem import AllChem, rdMolTransforms
from rdkit.Chem import Draw

import numpy as np
import mdtraj as md


psi_idx1, psi_idx2, psi_idx3, psi_idx4 = 10, 8, 14, 16
phi_idx1, phi_idx2, phi_idx3, phi_idx4 = 4, 6, 8, 10

def set_dihedral(mol, idx_1, idx2, idx3, idx4, angle):
    conf = mol.GetConformer()
    rdMolTransforms.SetDihedralDeg(conf, idx_1, idx2, idx3, idx4, angle)
    
    
state = "c5"
molecule = Chem.MolFromPDBFile(f'../../data/alanine/{state}.pdb', removeHs=False)

print("Genearting alanine dipeptide with various dihedral angles")
for psi in tqdm(np.linspace(-180, 180, 400, endpoint=False), desc="psi"):
    for phi in np.linspace(-180, 180, 400, endpoint=False):
        set_dihedral(molecule, psi_idx1, psi_idx2, psi_idx3, psi_idx4, psi)
        set_dihedral(molecule, phi_idx1, phi_idx2, phi_idx3, phi_idx4, phi)

        new_file_name = f"../../data/projection/alanine/{psi:.1f}_{phi:.1f}.pdb"
        if not os.path.exists(new_file_name):
            Chem.MolToPDBFile(molecule, new_file_name)
        else:
            print(f"File {new_file_name} already exists")
print("Done!!")