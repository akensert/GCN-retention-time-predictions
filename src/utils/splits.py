import numpy as np
import math
import multiprocessing
from collections import defaultdict
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from rdkit.Chem.Scaffolds import MurckoScaffold

from typing import Union, List, Tuple

from ops import transform_ops


def get_scaffold(string):
    mol = transform_ops.mol_from_string(string, True)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol, includeChirality=False)
    return scaffold

def train_valid_test_split(x: List,
                           y: Union[None, List] = None,
                           valid_frac: Union[None, float] = 0.2,
                           test_frac: Union[None, float] = 0.1,
                           shuffle: Union[None, bool] = True,
                           mode: str = 'random',
                           seed: int = 42) -> Tuple[List]:

    np.random.seed(seed)

    if mode == "index":
        # y are indices {1, 2, 3}
        return np.where(y == 1)[0], np.where(y == 2)[0], np.where(y == 3)[0]

    num_valid_examples = math.floor(
        (1 - test_frac / (test_frac + valid_frac))
        * (valid_frac + test_frac)
        * len(x)
    )

    num_test_examples = math.floor(
        test_frac / (test_frac + valid_frac)
        * (valid_frac + test_frac)
        * len(x)
    )

    if mode == 'scaffold':

        with multiprocessing.Pool(12) as pool:
            scaffolds = [i for i in pool.map(get_scaffold, x)]

        scaffold_groups = defaultdict(list)
        for index, scaffold in enumerate(scaffolds):
            scaffold_groups[scaffold].append(index)

        scaffold_sets = np.random.permutation(list(scaffold_groups.values()))

        train_indices, valid_indices, test_indices = [], [], []
        for scaffold_set in scaffold_sets:
            if len(test_indices) + len(scaffold_set) <= num_test_examples:
                test_indices.extend(scaffold_set)
            elif len(valid_indices) + len(scaffold_set) <= num_valid_examples:
                valid_indices.extend(scaffold_set)
            else:
                train_indices.extend(scaffold_set)

    else:
        train_indices, valid_indices = train_test_split(
            np.arange(len(x)),
            test_size=num_valid_examples+num_test_examples,
            shuffle=shuffle,
            random_state=seed,
            stratify=y if mode == 'stratify' else None
        )

        valid_indices, test_indices = train_test_split(
            valid_indices,
            test_size=num_test_examples,
            shuffle=shuffle,
            random_state=seed,
            stratify=y[valid_indices] if mode == 'stratify' else None
        )

    return train_indices, valid_indices, test_indices
