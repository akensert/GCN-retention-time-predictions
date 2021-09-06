import numpy as np
import pandas as pd
import multiprocessing
from functools import partial
from tqdm import tqdm
import os

from ops import transform_ops

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
                      use_counts,
                      valid_frac=0.1,
                      test_frac=0.1,
                      mode="random",
                      seed=42,
                      num_threads=12):

    dataframe = pd.read_csv(dataset_path)

    split_idx = dataframe.split_index.values.copy()

    train_idx, valid_idx, test_1_idx, test_2_idx = (
        np.where(split_idx == 1)[0], np.where(split_idx == 2)[0],
        np.where(split_idx == 3)[0], np.where(split_idx == 4)[0]
    )

    train_data = _get_dataset(
        dataframe.iloc[train_idx], bits, radius, use_counts, num_threads
    )
    valid_data = _get_dataset(
        dataframe.iloc[valid_idx], bits, radius, use_counts, num_threads
    )
    test_1_data = _get_dataset(
        dataframe.iloc[test_1_idx], bits, radius, use_counts, num_threads
    )

    if len(test_2_idx) > 0:
        test_2_data = _get_dataset(
            dataframe.iloc[test_2_idx], bits, radius, use_counts, num_threads
        )
    else:
        test_2_data = None
    return train_data, valid_data, test_1_data, test_2_data
