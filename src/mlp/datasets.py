import numpy as np
import pandas as pd
import multiprocessing
from functools import partial
from tqdm import tqdm

from utils import splits
from ops import transform_ops
import configuration


def _get_dataset(dataframe, bits, radius, use_counts, num_threads):
    data_keys = ['index', 'y', 'X']
    with multiprocessing.Pool(num_threads) as pool:
        ecfp = [
            e for e in pool.imap(
                partial(
                    transform_ops.ecfp_from_string,
                    bits=bits,
                    radius=radius,
                    use_counts=use_counts,
                ),
                dataframe.iloc[:, 2],
            )
    ]

    keep_idx = np.where(np.array(ecfp).sum(axis=1) > 0)[0]

    return {
        'index': np.array(dataframe.index)[keep_idx],
        'y': np.array(dataframe.iloc[:, 1])[keep_idx],
        'X': np.array(ecfp)[keep_idx]
    }

def get_ecfp_datasets(dataset_path,
                      bits,
                      radius,
                      use_counts=False,
                      valid_frac=0.1,
                      test_frac=0.1,
                      mode="random",
                      seed=42,
                      num_threads=12):

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

    train_data = _get_dataset(
        dataframe.iloc[train_idx], bits, radius, use_counts, num_threads
    )
    valid_data = _get_dataset(
        dataframe.iloc[valid_idx], bits, radius, use_counts, num_threads
    )
    test_data = _get_dataset(
        dataframe.iloc[test_idx], bits, radius, use_counts, num_threads
    )
    return train_data, valid_data, test_data
