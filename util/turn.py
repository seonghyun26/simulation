from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
from rdkit.Chem import Draw
import numpy as np

# Load your molecule (replace 'molecule.sdf' with your actual file)
org_state = "c5"
new_state = "alpha_L"
assert org_state != new_state, "Origin and new state should be different"
molecule = Chem.MolFromPDBFile(f'../data/alanine/{org_state}.pdb', removeHs=False)

# Convert to 3D
# AllChem.EmbedMolecule(molecule)
# AllChem.UFFOptimizeMolecule(molecule)

# Function to set dihedral angle
def set_dihedral(mol, idx1, idx2, idx3, idx4, angle):
    conf = mol.GetConformer()
    rdMolTransforms.SetDihedralDeg(conf, idx1, idx2, idx3, idx4, angle)

# Indices of the atoms defining the dihedrals (you need to find these for your specific molecule)
idx1, idx2, idx3, idx4 = 6, 8, 14, 16
idx5, idx6, idx7, idx8 = 4, 6, 8, 14

psi = rdMolTransforms.GetDihedralDeg(molecule.GetConformer(), idx1, idx2, idx3, idx4)
phi = rdMolTransforms.GetDihedralDeg(molecule.GetConformer(), idx5, idx6, idx7, idx8)
print("<--- Original dihedrals --->")
print(phi, psi)
print("<-------------------------->\n")

# for idx in [4, 6, 8, 14]:
#     print(molecule.GetAtomWithIdx(idx).GetSymbol())
#     print(molecule.GetConformer().GetAtomPosition(idx).x, molecule.GetConformer().GetAtomPosition(idx).y, molecule.GetConformer().GetAtomPosition(idx).z)


# Set new dihedrals
new_phi = 40
new_psi = 65
print("<--- Modified dihedrals --->")
print(new_phi, new_psi)
print("<-------------------------->")
set_dihedral(molecule, idx1, idx2, idx3, idx4, new_psi)
set_dihedral(molecule, idx5, idx6, idx7, idx8, new_phi)

# Save the new coordinates to a file, and print
Chem.MolToPDBFile(molecule, f'../data/alanine/{new_state}.pdb')
conf = molecule.GetConformer()
for i in range(molecule.GetNumAtoms()):
    pos = conf.GetAtomPosition(i)
    print(f'Atom {i}: {pos.x}, {pos.y}, {pos.z}')