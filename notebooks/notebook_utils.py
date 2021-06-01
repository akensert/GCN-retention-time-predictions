import numpy as np
import pandas as pd
import os

from mlp import datasets as ann_datasets
from ml import datasets as ml_datasets


def fit_and_predict_gnn(model_obj,
                        model_params,
                        model_weights,
                        datasets,
                        num_repl,
                        save_path):

    if not os.path.isdir('/'.join(save_path.split('/')[:-1])):
        os.makedirs('/'.join(save_path.split('/')[:-1]))

    batch_size = model_params['batch_size']
    num_epochs = model_params['num_epochs']
    del model_params['batch_size']
    del model_params['num_epochs']

    train_dataset, valid_dataset, test_dataset = datasets

    train_dataset.batch_size = batch_size
    valid_dataset.batch_size = batch_size
    test_dataset.batch_size = batch_size

    preds = {
        "train_indices": train_dataset.get_indices(),
        "valid_indices": valid_dataset.get_indices(),
        "test_indices": test_dataset.get_indices(),
        "train_trues": train_dataset.get_labels(),
        "valid_trues": valid_dataset.get_labels(),
        "test_trues": test_dataset.get_labels(),
        "train_preds": [],
        "valid_preds": [],
        "test_preds": [],
    }

    for _ in range(num_repl):

        model = model_obj(**model_params)

        if model_weights is not None:
            dummy_data = next(iter(train_dataset.get_iterator()))
            model([dummy_data['adjacency_matrix'], dummy_data['feature_matrix']])
            model.set_weights(model_weights)
        else:
            model.fit(
                train_dataset.get_iterator(),
                epochs=num_epochs, verbose=0
            )

        preds['train_preds'].append(
            model.predict(train_dataset.get_iterator())[1])
        preds['valid_preds'].append(
            model.predict(valid_dataset.get_iterator())[1])
        preds['test_preds'].append(
            model.predict(test_dataset.get_iterator())[1])

    preds['train_preds'] = np.array(preds['train_preds'])
    preds['valid_preds'] = np.array(preds['valid_preds'])
    preds['test_preds'] = np.array(preds['test_preds'])

    np.savez(save_path+'_output.npz', **preds)

def fit_and_predict_mlp(model_obj,
                        model_params,
                        model_weights,
                        datasets,
                        num_repl,
                        save_path):

    if not os.path.isdir('/'.join(save_path.split('/')[:-1])):
        os.makedirs('/'.join(save_path.split('/')[:-1]))

    batch_size = model_params['batch_size']
    num_epochs = model_params['num_epochs']
    bits = model_params['bits']
    radius = model_params['radius']

    del model_params['batch_size']
    del model_params['num_epochs']
    del model_params['bits']
    del model_params['radius']

    train, valid, test = ann_datasets.get_ecfp_datasets(
        f"../input/datasets/{save_path.split('/')[-2]}.csv",
        bits=bits, radius=radius,
    )

    preds = {
        "train_indices": train['index'],
        "valid_indices": valid['index'],
        "test_indices": test['index'],
        "train_trues": train['y'],
        "valid_trues": valid['y'],
        "test_trues": test['y'],
        "train_preds": [],
        "valid_preds": [],
        "test_preds": [],
    }

    for _ in range(num_repl):

        model = model_obj(**model_params)

        if model_weights is not None:
            model(train['X'][:1])
            model.set_weights(model_weights)
        else:
            model.fit(train['X'], train['y'],
                      batch_size=batch_size,
                      epochs=num_epochs)

        preds['train_preds'].append(
            model.predict(train['X'], train['y'])[1])
        preds['valid_preds'].append(
            model.predict(valid['X'], valid['y'])[1])
        preds['test_preds'].append(
            model.predict(test['X'], test['y'])[1])

    preds['train_preds'] = np.array(preds['train_preds'])
    preds['valid_preds'] = np.array(preds['valid_preds'])
    preds['test_preds'] = np.array(preds['test_preds'])

    np.savez(save_path+'_output.npz', **preds)

def fit_and_predict_ml(model_obj,
                       model_params,
                       model_best,
                       datasets,
                       num_repl,
                       save_path):


    if not os.path.isdir('/'.join(save_path.split('/')[:-1])):
        os.makedirs('/'.join(save_path.split('/')[:-1]))

    train, valid, test = ml_datasets.get_descriptor_datasets(
        f"../input/datasets/{save_path.split('/')[-2]}.csv"
    )

    preds = {
        "train_indices": train['index'],
        "valid_indices": valid['index'],
        "test_indices": test['index'],
        "train_trues": train['y'],
        "valid_trues": valid['y'],
        "test_trues": test['y'],
        "train_preds": [],
        "valid_preds": [],
        "test_preds": [],
    }

    for _ in range(num_repl):

        if model_best is not None:
            model = model_best
        else:
            model = model_obj(**model_params)
            model.fit(train['X'], train['y'])

        preds['train_preds'].append(model.predict(train['X']))
        preds['valid_preds'].append(model.predict(valid['X']))
        preds['test_preds'].append(model.predict(test['X']))

    preds['train_preds'] = np.array(preds['train_preds'])
    preds['valid_preds'] = np.array(preds['valid_preds'])
    preds['test_preds'] = np.array(preds['test_preds'])

    np.savez(save_path+'_output.npz', **preds)
