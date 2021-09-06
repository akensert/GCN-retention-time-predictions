import numpy as np
import pandas as pd
import multiprocessing
from tqdm import tqdm
from functools import partial
import os

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog('rdApp.*')

from ops import transform_ops


def _get_dataset(dataframe, num_threads=12):
    data_keys = ['index', 'y', 'X']
    with multiprocessing.Pool(num_threads) as pool:
        desc = [
            d for d in pool.imap(
                    partial(
                        transform_ops.desc_from_string,
                        thresh=None
                    ),
                    dataframe.iloc[:, 2],
                )
        ]

    keep_idx = np.where(np.array(desc).sum(axis=1) > 0)[0]

    return {
        'index': np.array(dataframe.index)[keep_idx],
        'y': np.array(dataframe.iloc[:, 1])[keep_idx],
        'X': np.array(desc)[keep_idx]
    }

def get_descriptor_datasets(dataset_path='../input/datasets/SMRT.csv'):

    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    dataframe = pd.read_csv(dataset_path)

    split_idx = dataframe.split_index.values.copy()

    train_idx, valid_idx, test_1_idx, test_2_idx = (
        np.where(split_idx == 1)[0], np.where(split_idx == 2)[0],
        np.where(split_idx == 3)[0], np.where(split_idx == 4)[0]
    )

    train_data = _get_dataset(dataframe.iloc[train_idx])
    valid_data = _get_dataset(dataframe.iloc[valid_idx])
    test_1_data = _get_dataset(dataframe.iloc[test_1_idx])
    if len(test_2_idx) > 0:
        test_2_data = _get_dataset(dataframe.iloc[test_2_idx])
    else:
        test_2_data = None

    keep_idx = np.where(train_data['X'].std(axis=0) != 0)[0]
    train_data['X'] = train_data['X'][:, keep_idx]
    valid_data['X'] = valid_data['X'][:, keep_idx]
    test_1_data['X'] = test_1_data['X'][:, keep_idx]
    if test_2_data:
        test_2_data['X'] = test_2_data['X'][:, keep_idx]

    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(train_data['X'])
    train_data['X'] = scaling.transform(train_data['X'])
    valid_data['X'] = scaling.transform(valid_data['X'])
    test_1_data['X'] = scaling.transform(test_1_data['X'])
    if test_2_data:
        test_2_data['X'] = scaling.transform(test_2_data['X'])
    return train_data, valid_data, test_1_data, test_2_data
