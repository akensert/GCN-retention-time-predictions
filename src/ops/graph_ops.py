import tensorflow as tf
import numpy as np
from rdkit import Chem

from . import feature_ops


def get_edge_dim():
    """Hacky way to get edge dim from bond_featurizer"""
    mol = Chem.MolFromSmiles('CC')
    return len(feature_ops.bond_featurizer(mol.GetBonds()[0]))

def get_node_dim():
    """Hacky way to get node dim from atom_featurizer"""
    mol = Chem.MolFromSmiles('CC')
    return len(feature_ops.atom_featurizer(mol.GetAtoms()[0]))

def get_adjacency_matrix(mol):
    return Chem.GetAdjacencyMatrix(mol)

def get_edge_tensor(mol):
    feature_dim = get_edge_dim()
    edge_features = np.zeros(
        shape=(feature_dim, mol.GetNumAtoms(), mol.GetNumAtoms()),
        dtype='float32'
    )
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        for neighbor in atom.GetNeighbors():
            j = neighbor.GetIdx()
            bond = mol.GetBondBetweenAtoms(i, j)
            bond_features = feature_ops.bond_featurizer(bond)
            edge_features[:, [i, j], [j, i]] = bond_features[:, np.newaxis]
    return edge_features

def get_node_features(mol):
    node_features = np.array([
        feature_ops.atom_featurizer(atom) for atom in mol.GetAtoms()
    ], dtype='float32')
    return node_features

def dense_to_sparse(arr):
    indices = tf.cast(tf.where(arr != 0), tf.int64)
    values = tf.cast(tf.gather_nd(arr, indices), tf.float32)
    shape = tf.cast(tf.shape(arr), tf.int64)
    return indices.numpy(), values.numpy(), shape.numpy()
