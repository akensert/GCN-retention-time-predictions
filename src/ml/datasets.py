import numpy as np
import pandas as pd
import multiprocessing
from tqdm import tqdm
from functools import partial

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog('rdApp.*')

from utils import splits
from ops import transform_ops
import configuration


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
    dataset_name = dataset_path.split('/')[-1].split('.')[0]

    X = dataframe.iloc[:, 2].values.copy()
    if configuration.datasets[dataset_name]["mode"] == "index":
        y = dataframe.split_index.values.copy()
    else:
        y = dataframe.iloc[:, 1].values.copy()

    train_idx, valid_idx, test_idx = splits.train_valid_test_split(
        x=X, # smiles/inchi
        y=y, # labels (e.g. retention time)
        valid_frac=configuration.datasets[dataset_name]["valid_frac"],
        test_frac=configuration.datasets[dataset_name]["test_frac"],
        mode=configuration.datasets[dataset_name]["mode"],
        seed=configuration.seeds["splits"])

    train_data = _get_dataset(dataframe.iloc[train_idx])
    valid_data = _get_dataset(dataframe.iloc[valid_idx])
    test_data = _get_dataset(dataframe.iloc[test_idx])

    keep_idx = np.where(train_data['X'].std(axis=0) != 0)[0]
    train_data['X'] = train_data['X'][:, keep_idx]
    valid_data['X'] = valid_data['X'][:, keep_idx]
    test_data['X'] = test_data['X'][:, keep_idx]

    scaling = StandardScaler().fit(train_data['X'])
    train_data['X'] = scaling.transform(train_data['X'])
    valid_data['X'] = scaling.transform(valid_data['X'])
    test_data['X'] = scaling.transform(test_data['X'])

    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(train_data['X'])
    train_data['X'] = scaling.transform(train_data['X'])
    valid_data['X'] = scaling.transform(valid_data['X'])
    test_data['X'] = scaling.transform(test_data['X'])

    return train_data, valid_data, test_data
