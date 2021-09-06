import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
from abc import ABCMeta, abstractmethod

from utils import metrics


class BaseModel(tf.keras.Model, metaclass=ABCMeta):

    @abstractmethod
    def __init__(self,
                 loss_fn,
                 optimizer,
                 initial_learning_rate,
                 end_learning_rate,
                 power,
                 **kwargs):
        """
        In parent class, inherit from BaseModel and run its __init__.
        Then define the different layers and hyperparameters that will be
        used in the call method.
        """
        super(BaseModel, self).__init__(**kwargs)

        self.initial_learning_rate = initial_learning_rate
        self.end_learning_rate = end_learning_rate              # NOT USED
        self.power = power                                      # NOT USED
        self.compile(loss=loss_fn(), optimizer=optimizer(initial_learning_rate))

    @abstractmethod
    def call(self, inputs, training):
        """
        The computations for the forward pass should go here for the child class
        """
        return inputs

    def _train_step(self, inputs, training=True):
        """
        Performs a forward pass and a backward pass
        """
        A = inputs['adjacency_matrix']
        H = inputs['feature_matrix']
        y = inputs['label']
        with tf.GradientTape() as tape:
            y_pred = self(inputs=[A, H], training=training)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def _predict_step(self, data, training=False):
        """
        Performs a forward pass only
        """
        A = data['adjacency_matrix']
        H = data['feature_matrix']
        y_pred = self(inputs=[A, H], training=training)
        return y_pred

    def fit(self, train_dataset, additional_datasets=None,
            epochs=1, steps_per_epoch=None, verbose=0):

        if steps_per_epoch is not None:
            num_batches = steps_per_epoch
            train_dataset = train_dataset.take(num_batches)
        elif verbose:
            num_batches = train_dataset.reduce(np.int64(0), lambda x, _: x + 1).numpy()

        if additional_datasets is not None:
            best_error = float('inf')
            self.learning_curves = {}
            for name in ['train'] + list(additional_datasets.keys()):
                self.learning_curves[name + '_loss'] = []
                self.learning_curves[name + '_mae'] = []
                self.learning_curves[name + '_mre'] = []
                self.learning_curves[name + '_rmse'] = []
                self.learning_curves[name + '_r2'] = []

        for epoch in range(epochs):

            if verbose:
                train_dataset = tqdm(train_dataset, total=num_batches)

            for batch in train_dataset:

                result = self._train_step(batch)

                if verbose:
                    current_lr = self.optimizer._decayed_lr(tf.float32).numpy()
                    description  = f'epoch {epoch:03d} : '
                    description += f'lr {current_lr:.6f} : '
                    description += f'loss {result["loss"]:5.3f} : '
                    if epoch != 0 and additional_datasets is not None:
                        description += f'valid {self.learning_curves["valid_mae"][-1]:6.3f} '
                    train_dataset.set_description(description)

            if additional_datasets is not None:

                trues, preds = self.predict(train_dataset, verbose=0)
                self._accumulate_learning_curve(trues, preds, 'train')

                for name, dataset in additional_datasets.items():

                    trues, preds = self.predict(dataset, verbose=0)
                    self._accumulate_learning_curve(trues, preds, name)

                if self.learning_curves["valid_loss"][-1] < best_error:
                    best_error = self.learning_curves["valid_loss"][-1]
                    self.best_weights = self.get_weights()

            # decay learning rate by a factor of 0.1 for last 20% of training
            if epoch == int(epochs * 0.8):
                decayed_lr = (self.initial_learning_rate * 0.1)
                self.optimizer.learning_rate.assign(decayed_lr)

        return self

    def predict(self, test_dataset, verbose=0):

        if verbose:
            num_batches = test_dataset.reduce(np.int64(0), lambda x, _: x + 1)
            test_dataset = tqdm(test_dataset, total=num_batches.numpy())

        preds, trues = [], []
        for batch in test_dataset:
            y_pred = self._predict_step(batch)
            preds.append(np.squeeze(y_pred, axis=-1))
            trues.append(np.squeeze(batch['label'], axis=-1))

        return np.concatenate(trues, axis=0), np.concatenate(preds, axis=0)

    def get_latent_spaces(self, test_dataset):
        latent_space = []
        for batch in  test_dataset:
            A = batch['adjacency_matrix']
            H = batch['feature_matrix']
            out = self.latent(inputs=[A, H])
            latent_space.append(out)
        return np.concatenate(latent_space, axis=0)

    def _accumulate_learning_curve(self, trues, preds, name):
        loss = self.compiled_loss(
            tf.convert_to_tensor(trues), tf.convert_to_tensor(preds))
        mae = metrics.get('mae')(trues, preds)
        mre = metrics.get('mre')(trues, preds)
        rmse = metrics.get('rmse')(trues, preds)
        r2 = metrics.get('r2')(trues, preds)
        self.learning_curves[f'{name}_loss'].append(loss.numpy())
        self.learning_curves[f'{name}_mae'].append(mae)
        self.learning_curves[f'{name}_mre'].append(mre)
        self.learning_curves[f'{name}_rmse'].append(rmse)
        self.learning_curves[f'{name}_r2'].append(r2)
