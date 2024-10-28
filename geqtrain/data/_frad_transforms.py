from rdkit.Chem import rdMolTransforms
import copy
import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D
from rdkit.Chem import AllChem
import torch

# import pandas as pd
# from rdkit.Chem import rdmolops
# from rdkit.Chem.rdchem import BondType as BT
# from rdkit.Chem.rdchem import HybridizationType
# from torch_geometric.utils import one_hot, scatter
# from numpy import load

# from torch_geometric.data import (
#     Data,
#     InMemoryDataset,
#     download_url,
#     extract_zip,
# )
# from rdkit.Chem import Draw
# from collections import defaultdict
# import collections
# import random
# import math
# import py3Dmol
# from io import BytesIO
# from PIL import Image
# import torch

def get_torsions(mol_list):
    """
    Extracts the torsion angles (dihedrals) from a list of molecules.

    This function identifies all the torsion angles in the given list of molecules and returns a list of these torsions.
    A torsion angle is defined by four atoms and is calculated based on the connectivity of these atoms in the molecule.

    Args:
        mol_list (list): A list of RDKit molecule objects.

    Returns:
        list: A list of tuples, where each tuple contains four integers representing the indices of the atoms
              that define a torsion angle in the molecule.

    """
    atom_counter = 0
    torsionList = []
    for m in mol_list:
        torsionSmarts = '[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]'
        torsionQuery = Chem.MolFromSmarts(torsionSmarts)
        matches = m.GetSubstructMatches(torsionQuery)
        for match in matches:
            idx2 = match[0]
            idx3 = match[1]
            bond = m.GetBondBetweenAtoms(idx2, idx3)
            jAtom = m.GetAtomWithIdx(idx2)
            kAtom = m.GetAtomWithIdx(idx3)
            for b1 in jAtom.GetBonds():
                if (b1.GetIdx() == bond.GetIdx()):
                    continue
                idx1 = b1.GetOtherAtomIdx(idx2)
                for b2 in kAtom.GetBonds():
                    if ((b2.GetIdx() == bond.GetIdx())
                            or (b2.GetIdx() == b1.GetIdx())):
                        continue
                    idx4 = b2.GetOtherAtomIdx(idx3)
                    # skip 3-membered rings
                    if (idx4 == idx1):
                        continue
                    # skip torsions that include hydrogens
                    #                     if ((m.GetAtomWithIdx(idx1).GetAtomicNum() == 1)
                    #                         or (m.GetAtomWithIdx(idx4).GetAtomicNum() == 1)):
                    #                         continue
                    if m.GetAtomWithIdx(idx4).IsInRing():
                        torsionList.append(
                            (idx4 + atom_counter, idx3 + atom_counter, idx2 + atom_counter, idx1 + atom_counter))
                        break
                    else:
                        torsionList.append(
                            (idx1 + atom_counter, idx2 + atom_counter, idx3 + atom_counter, idx4 + atom_counter))
                        break
                break

        atom_counter += m.GetNumAtoms()
    return torsionList

def SetDihedral(conf, atom_idx, new_vale):
    """
    Sets the value of a dihedral angle (torsion) in a molecule's conformation.

    This function modifies the dihedral angle defined by four atoms in the given molecule conformation to the specified value.

    Args:
        conf (RDKit Conformer): The conformation of the molecule.
        atom_idx (tuple): A tuple of four integers representing the indices of the atoms that define the dihedral angle.
        new_vale (float): The new value of the dihedral angle in degrees.

    """
    try:
        rdMolTransforms.SetDihedralDeg(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3], new_vale)
    except:
        rdMolTransforms.SetDihedralDeg(conf, atom_idx[1], atom_idx[0], atom_idx[2], atom_idx[3], new_vale)

def GetDihedral(conf, atom_idx):
    """
    Retrieves the value of a dihedral angle (torsion) in a molecule's conformation.

    This function returns the current value of the dihedral angle defined by four atoms in the given molecule conformation.

    Args:
        conf (RDKit Conformer): The conformation of the molecule.
        atom_idx (tuple): A tuple of four integers representing the indices of the atoms that define the dihedral angle.

    Returns:
        float: The value of the dihedral angle in degrees.

    """
    return rdMolTransforms.GetDihedralDeg(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3])

def apply_changes(mol, values, rotable_bonds):
    """
    Applies specified dihedral angle changes to a molecule based on the provided values for the dihedral angles.

    Args:
        mol (RDKit Mol): The original molecule to which the changes will be applied.
        values (list of float): A list of new values for the dihedral angles in degrees.
        rotable_bonds (list of tuple): A list of tuples, where each tuple contains four integers representing
                                       the indices of the atoms that define a rotatable bond.

    Returns:
        None: modifies inplace the input mol (RDKit Mol)
    """
    [SetDihedral(mol.GetConformer(), rotable_bonds[r], values[r].item()) for r in range(len(rotable_bonds))]

def transform_noise(data, position_noise_scale):
    return data + np.random.normal(loc=0, scale=1, size=data.shape) * position_noise_scale

def nosify_mol(data):
    '''
    data: pyg data object
    return: pyg data object with new coords
    '''
    dihedral_noise_tau, coords_noise_tau = 2, 0.04
    mol = Chem.MolFromSmiles(str(data.smiles))
    mol = Chem.AddHs(mol)
    id:int = AllChem.EmbedMolecule(mol, useRandomCoords=True, enforceChirality = True, useExpTorsionAnglePrefs= True, useBasicKnowledge= True, useMacrocycleTorsions= True)
    # if id == -1: return data

    # enforceChirality : enforce the correct chirality if chiral centers are present.
    # useExpTorsionAnglePrefs : impose experimental torsion angle preferences
    # useBasicKnowledge : impose basic knowledge such as flat rings
    # printExpTorsionAngles : print the output from the experimental torsion angles
    # useMacrocycleTorsions : use additional torsion profiles for macrocycles
    #! https://greglandrum.github.io/rdkit-blog/posts/2024-07-28-confgen-and-intramolecular-hbonds.html
    #! https://www.rdkit.org/docs/RDKit_Book.html#conformer-generation
    # EmbedMolecule doc
    # https://www.rdkit.org/docs/source/rdkit.Chem.rdDistGeom.html#rdkit.Chem.rdDistGeom.EmbedMolecule

    # https://www.rdkit.org/docs/source/rdkit.Chem.rdForceFieldHelpers.html
    # AllChem.MMFFOptimizeMolecule(mol)

    # https://www.rdkit.org/docs/source/rdkit.Chem.AllChem.html
    # https://www.rdkit.org/docs/GettingStartedInPython.html#working-with-3d-molecules

    #? https://www.rdkit.org/docs/source/rdkit.Chem.rdDistGeom.html#rdkit.Chem.rdDistGeom.EmbedMolecule

    # rotable_bonds = get_torsions([mol]) #! this MUST BE EXTRACT AT DSET LVL
    # dihedral_angles_degrees = np.array([GetDihedral(mol.GetConformer(), rot_bond) for rot_bond in rotable_bonds]) #! this MUST BE EXTRACT AT DSET LVL

    rotable_bonds = data.rotable_bonds.tolist()
    if rotable_bonds:
        original_dihedral_angles_degrees = data.dihedral_angles_degrees
        # apply dihedral noise
        noised_dihedral_angles_degrees = original_dihedral_angles_degrees + np.random.normal(loc=0, scale=1, size=original_dihedral_angles_degrees.shape) * dihedral_noise_tau
        apply_changes(mol, noised_dihedral_angles_degrees, rotable_bonds)

    # apply coords noise
    pos_after_dihedral_noise = mol.GetConformer().GetPositions()
    pos_noise_to_be_predicted = np.random.normal(loc=0, scale=1, size=pos_after_dihedral_noise.shape) * coords_noise_tau

    data.noise_target = torch.tensor(pos_noise_to_be_predicted, dtype=torch.float) # todo, which field to set?
    data.pos = torch.tensor(pos_after_dihedral_noise + pos_noise_to_be_predicted, dtype=torch.float)

    return data


# def nosify_mol(data):
#     '''
#     data: pyg data object
#     return: pyg data object with new coords
#     '''
#     # cast to mol
#     mol = Chem.MolFromSmiles(str(data.smiles))
#     mol = Chem.AddHs(mol)

#     # apply dihedral noise
#     mol_tmp = apply_dihedral_noise(mol)
#     mol_tmp = Chem.AddHs(mol_tmp)

#     # get new atoms coords
#     coord_conf = mol_tmp.GetConformer()
#     pos = coord_conf.GetPositions()
#     pos = torch.tensor(pos, dtype=torch.float)
#     data.pos = pos # and set it into data object

#     # write assert here about different angles

#     # apply coord noise
#     data = apply_coord_noise(data) # this already sets noisified coords and sets pos_target into data obj

#     return data

