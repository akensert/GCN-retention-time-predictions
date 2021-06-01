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

    def _train_step(self, inputs):
        """
        Performs a forward pass and a backward pass
        """
        A = inputs['adjacency_matrix']
        H = inputs['feature_matrix']
        y = inputs['label']
        with tf.GradientTape() as tape:
            y_pred = self(inputs=[A, H], training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def _predict_step(self, data):
        """
        Performs a forward pass only
        """
        A = data['adjacency_matrix']
        H = data['feature_matrix']
        y_pred = self(inputs=[A, H], training=False)
        return y_pred

    def fit(self, train_dataset, valid_dataset=None, epochs=1, steps_per_epoch=None, verbose=0):

        if steps_per_epoch is not None:
            num_batches = steps_per_epoch
            train_dataset = train_dataset.take(num_batches)
        elif verbose:
            num_batches = train_dataset.reduce(np.int64(0), lambda x, _: x + 1).numpy()

        if valid_dataset is not None:
            best_valid_loss = np.inf
            self.valid_losses = []

        for epoch in range(epochs):

            if valid_dataset is not None:
                trues, preds = self.predict(valid_dataset, verbose=0)
                valid_loss = self.compiled_loss(
                    tf.convert_to_tensor(trues), tf.convert_to_tensor(preds)
                )
                # valid_loss = tf.convert_to_tensor(
                #     metrics.get('rmse')(trues, preds)
                # )
                self.valid_losses.append(valid_loss.numpy())
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    self.best_weights = self.get_weights()

            if verbose:
                train_dataset = tqdm(train_dataset, total=num_batches)

            for batch in train_dataset:

                result = self._train_step(batch)

                if verbose:
                    current_lr = self.optimizer._decayed_lr(tf.float32).numpy()
                    description  = f'epoch {epoch:03d} : '
                    description += f'lr {current_lr:.6f} : '
                    description += f'loss {result["loss"]:5.3f} : '
                    if valid_dataset is not None:
                        description += f'valid {valid_loss:6.3f} '
                    train_dataset.set_description(description)

            # decay learning rate by a factor of 0.1 for last 20% of training
            if epoch >= (epochs * 0.8):
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
