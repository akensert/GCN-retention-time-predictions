import numpy as np
from typing import Union, List
from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem import Lipinski
from rdkit.Chem import Crippen
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdPartialCharges


atom_features = [
    'chiral_center',
    'cip_code',
    'crippen_log_p_contrib',
    'crippen_molar_refractivity_contrib',
    'degree',
    'element',
    'formal_charge',
    'gasteiger_charge',
    'hybridization',
    'is_aromatic',
    'is_h_acceptor',
    'is_h_donor',
    'is_hetero',
    'is_in_ring_size_n',
    'labute_asa_contrib',
    'mass',
    'num_hs',
    'num_radical_electrons',
    'num_valence',
    'tpsa_contrib',
]

bond_features = [
    'bondstereo',
    'bondtype',
    'is_conjugated',
    'is_in_ring',
    'is_rotatable',
]


def onehot_encode(x: Union[float, int, str],
                  allowable_set: List[Union[float, int, str]]) -> List[float]:
    return list(map(lambda s: float(x == s), allowable_set))

def encode(x: Union[float, int, str]) -> List[float]:
    if x is None or np.isnan(x):
        x = 0.0
    return [float(x)]

def bond_featurizer(bond: Chem.Bond) -> np.ndarray:
    return np.concatenate([
        globals()[bond_feature](bond) for bond_feature in bond_features
    ], axis=0)

def atom_featurizer(atom: Chem.Atom) -> np.ndarray:
    return np.concatenate([
        globals()[atom_feature](atom) for atom_feature in atom_features
    ], axis=0)

def bondtype(bond: Chem.Bond) -> List[float]:
    return onehot_encode(
        x=bond.GetBondType(),
        allowable_set=[
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC
        ]
    )

def is_in_ring(bond: Chem.Bond) -> List[float]:
    return encode(
        x=bond.IsInRing()
    )

def is_conjugated(bond):
    return encode(
        x=bond.GetIsConjugated()
    )

def is_rotatable(bond: Chem.Bond) -> List[float]:
    mol = bond.GetOwningMol()
    atom_indices = tuple(
        sorted([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]))
    return encode(
        x=atom_indices in Lipinski._RotatableBonds(mol)
    )

def bondstereo(bond: Chem.Bond) -> List[float]:
    return onehot_encode(
        x=bond.GetStereo(),
        allowable_set=[
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
            Chem.rdchem.BondStereo.STEREOANY,
        ]
    )

def element(atom: Chem.Atom) -> List[float]:
    return onehot_encode(
        x=atom.GetSymbol(),
        allowable_set=[
            'H',  'He', 'Li', 'Be', 'B',  'C',  'N',  'O',  'F',  'Ne',
            'Na', 'Mg', 'Al', 'Si', 'P',  'S',  'Cl', 'Ar', 'K',  'Ca',
            'Sc', 'Ti', 'V',  'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y',  'Zr',
            'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
            'Sb', 'Te', 'I',  'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
            'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
            'Lu', 'Hf', 'Ta', 'W',  'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
            'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
            'Pa', 'U',  'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
            'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
            'Rg', 'Cn'
        ]
    )

def hybridization(atom: Chem.Atom) -> List[float]:
    return onehot_encode(
        x=atom.GetHybridization(),
        allowable_set=[
            Chem.rdchem.HybridizationType.S,
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
        ]
    )

def cip_code(atom: Chem.Atom) -> List[float]:
    if atom.HasProp("_CIPCode"):
        return onehot_encode(
            x=atom.GetProp("_CIPCode"),
            allowable_set=[
                "R", "S"
            ]
        )
    return [0.0, 0.0]

def chiral_center(atom: Chem.Atom) -> List[float]:
    return encode(
        x=atom.HasProp("_ChiralityPossible")
    )

def formal_charge(atom: Chem.Atom) -> List[float]:
    return onehot_encode(
        x=min(max(atom.GetFormalCharge(), -1), 1),
        allowable_set=[-1, 1, 1]
    )

def mass(atom: Chem.Atom) -> List[float]:
    return encode(
        x=atom.GetMass() / 100
    )

def num_hs(atom: Chem.Atom) -> List[float]:
    return onehot_encode(
        x=min(atom.GetTotalNumHs(), 4),
        allowable_set=[0, 1, 2, 3, 4]
    )

def num_valence(atom: Chem.Atom) -> List[float]:
    return onehot_encode(
        x=min(atom.GetTotalValence(), 6),
        allowable_set=[0, 1, 2, 3, 4, 5, 6])

def degree(atom: Chem.Atom) -> List[float]:
    return onehot_encode(
        x=min(atom.GetDegree(), 5),
        allowable_set=[0, 1, 2, 3, 4, 5]
    )

def is_aromatic(atom: Chem.Atom) -> List[float]:
    return encode(
        x=atom.GetIsAromatic()
    )

def is_hetero(atom: Chem.Atom) -> List[float]:
    mol = atom.GetOwningMol()
    return encode(
        x=atom.GetIdx() in [i[0] for i in Lipinski._Heteroatoms(mol)]
    )

def is_h_donor(atom: Chem.Atom) -> List[float]:
    mol = atom.GetOwningMol()
    return encode(
        x=atom.GetIdx() in [i[0] for i in Lipinski._HDonors(mol)]
    )

def is_h_acceptor(atom: Chem.Atom) -> List[float]:
    mol = atom.GetOwningMol()
    return encode(
        x=atom.GetIdx() in [i[0] for i in Lipinski._HAcceptors(mol)]
    )

def is_in_ring_size_n(atom: Chem.Atom) -> List[float]:
    for ring_size in [10, 9, 8, 7, 6, 5, 4, 3, 0]:
        if atom.IsInRingSize(ring_size): break
    return onehot_encode(
        x=ring_size,
        allowable_set=[0, 3, 4, 5, 6, 7, 8, 9, 10]
    )

def num_radical_electrons(atom: Chem.Atom) -> List[float]:
    num_radical_electrons = atom.GetNumRadicalElectrons()
    return onehot_encode(
        x=min(num_radical_electrons, 2),
        allowable_set=[0, 1, 2]
    )

def crippen_log_p_contrib(atom: Chem.Atom) -> List[float]:
    mol = atom.GetOwningMol()
    return encode(
        x=Crippen._GetAtomContribs(mol)[atom.GetIdx()][0]
    )

def crippen_molar_refractivity_contrib(atom: Chem.Atom) -> List[float]:
    mol = atom.GetOwningMol()
    return encode(
        x=Crippen._GetAtomContribs(mol)[atom.GetIdx()][1]
    )

def tpsa_contrib(atom: Chem.Atom) -> List[float]:
    mol = atom.GetOwningMol()
    return encode(
        x=rdMolDescriptors._CalcTPSAContribs(mol)[atom.GetIdx()]
    )

def labute_asa_contrib(atom: Chem.Atom) -> List[float]:
    mol = atom.GetOwningMol()
    return encode(
        x=rdMolDescriptors._CalcLabuteASAContribs(mol)[0][atom.GetIdx()]
    )

def gasteiger_charge(atom: Chem.Atom) -> List[float]:
    mol = atom.GetOwningMol()
    rdPartialCharges.ComputeGasteigerCharges(mol)
    return encode(
        x=atom.GetDoubleProp('_GasteigerCharge')
    )
