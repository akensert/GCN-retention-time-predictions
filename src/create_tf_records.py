import tensorflow as tf
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from tqdm import tqdm
import multiprocessing
import argparse
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from ops import transform_ops
from ops import graph_ops
from utils import splits
import configuration

tf.config.set_visible_devices([], 'GPU')

RDLogger.DisableLog('rdApp.*')


def get_graph(data):

    index, label, string = data

    mol = transform_ops.mol_from_string(string, catch_errors=False)

    if mol is None:
        return None

    node_features = graph_ops.get_node_features(mol)
    node_min = np.min(node_features, axis=(0))
    node_max = np.max(node_features, axis=(0))

    adjacency_matrix = graph_ops.get_adjacency_matrix(mol)

    edge_tensor = graph_ops.get_edge_tensor(mol)
    edge_min = np.min(edge_tensor, axis=(1, 2))
    edge_max = np.max(edge_tensor, axis=(1, 2))

    (findices, fvalues, fshape) = graph_ops.dense_to_sparse(node_features)
    (aindices, avalues, ashape) = graph_ops.dense_to_sparse(adjacency_matrix)
    (eindices, evalues, eshape) = graph_ops.dense_to_sparse(edge_tensor)

    return (
        index, label, string,
        findices, fvalues, fshape, node_min, node_max,
        aindices, avalues, ashape,
        eindices, evalues, eshape, edge_min, edge_max
    )

def create_tfrecords(dataframes, save_paths, num_threads):

    def _serialize_bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        value = tf.io.serialize_tensor(value)
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    if not os.path.isdir('/'.join(save_paths[0].split('/')[:-1])):
        os.makedirs('/'.join(save_paths[0].split('/')[:-1]))

    graph_data_keys = [
        'index',    'label', 'string',
        'findices', 'fvalues', 'fshape', 'fmin', 'fmax',
        'aindices', 'avalues', 'ashape',
        'eindices', 'evalues', 'eshape', 'emin', 'emax'
    ]

    for dataframe, save_path in zip(dataframes, save_paths):

        with multiprocessing.Pool(num_threads) as pool:
            data = dict(zip(graph_data_keys, zip(*[
                i for i in tqdm(
                    pool.imap(get_graph, zip(
                        dataframe.index,
                        dataframe.iloc[:, 1],
                        dataframe.iloc[:, 2])),
                    total=len(dataframe),)
                if i is not None])))

        if save_path.endswith('train.tfrec'):
            fmin = np.min(data['fmin'], axis=0)
            fmax = np.max(data['fmax'], axis=0)
            emin = np.min(data['emin'], axis=0)
            emax = np.max(data['emax'], axis=0)

        with tf.io.TFRecordWriter(save_path) as writer:

            for i in range(len(data['label'])):

                index = data['index'][i]
                label = data['label'][i]
                string = data['string'][i]
                findices = data['findices'][i]
                fvalues = data['fvalues'][i]
                fshape = data['fshape'][i]
                aindices = data['aindices'][i]
                avalues = data['avalues'][i]
                ashape = data['ashape'][i]
                eindices = data['eindices'][i]
                evalues = data['evalues'][i]
                eshape = data['eshape'][i]

                example_proto = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'index': _serialize_bytes_feature(np.array(index, dtype='int64')),
                            'label': _serialize_bytes_feature(np.array(label, dtype='float32')),
                            'string': _serialize_bytes_feature(string),

                            'feature_indices': _serialize_bytes_feature(findices),
                            'feature_values': _serialize_bytes_feature(fvalues),
                            'feature_shape': _serialize_bytes_feature(fshape),
                            'feature_min': _serialize_bytes_feature(fmin),
                            'feature_max': _serialize_bytes_feature(fmax),

                            'adjacency_indices': _serialize_bytes_feature(aindices),
                            'adjacency_values': _serialize_bytes_feature(avalues),
                            'adjacency_shape': _serialize_bytes_feature(ashape),

                            'edge_indices': _serialize_bytes_feature(eindices),
                            'edge_values': _serialize_bytes_feature(evalues),
                            'edge_shape': _serialize_bytes_feature(eshape),
                            'edge_min': _serialize_bytes_feature(emin),
                            'edge_max': _serialize_bytes_feature(emax),
                        }
                    )
                )

                writer.write(example_proto.SerializeToString())


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        default='../input/datasets/SMRT.csv')
    parser.add_argument('--num_threads', type=int, default=4)
    args = parser.parse_args()

    if args.dataset_path.endswith('*'):
        import glob
        dataset_paths = glob.glob(args.dataset_path)
    else:
        dataset_paths = [args.dataset_path]

    for dataset_path in dataset_paths:

        dataset_name = dataset_path.split('/')[-1].split('.')[0]

        dataframe = pd.read_csv(dataset_path)

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

        create_tfrecords(
            dataframes=[
                dataframe.iloc[train_idx],
                dataframe.iloc[valid_idx],
                dataframe.iloc[test_idx]
            ],
            save_paths = [
                '../input/tfrecords/' + dataset_name + '/train.tfrec',
                '../input/tfrecords/' + dataset_name + '/valid.tfrec',
                '../input/tfrecords/' + dataset_name + '/test.tfrec'
            ],
             num_threads=args.num_threads)
