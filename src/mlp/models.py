import tensorflow as tf
import numpy as np
from tqdm import tqdm

from utils import metrics

class MLPModel(tf.keras.Model):

    def __init__(self,
                 hidden_units=[256,],
                 kernel_regularizer=None,
                 dropout_rate=0.0,
                 initial_learning_rate=1e-3,
                 end_learning_rate=5e-5, # NOT USED
                 power=2.0, # NOT USED
                 loss_fn=tf.keras.losses.Huber,
                 optimizer=tf.keras.optimizers.Adam,
                 **kwargs):

        super().__init__(**kwargs)

        self.initial_learning_rate = initial_learning_rate

        self.seq_model = tf.keras.Sequential()
        for units in hidden_units:
            self.seq_model.add(tf.keras.layers.Dense(
                units=units,
                kernel_regularizer=kernel_regularizer,
                activation='relu'
            ))
            self.seq_model.add(tf.keras.layers.Dropout(dropout_rate))

        self.seq_model.add(tf.keras.layers.Dense(
            units=1,
            kernel_regularizer=kernel_regularizer,
            dtype='float32'))

        self.compile(loss=loss_fn(), optimizer=optimizer(initial_learning_rate))

    def summary(self):
        return self.seq_model.summary()

    def call(self, inputs, training):
        return self.seq_model(inputs)

    def _train_step(self, inputs):
        X, y = inputs
        with tf.GradientTape() as tape:
            y_pred = self(X, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def _predict_step(self, X):
        return self(X, training=False)

    def fit(self, X, y, additional_datasets=None, batch_size=32,
            epochs=1, steps_per_epoch=None, shuffle=True, verbose=0):

        train_dataset = tf.data.Dataset.from_tensor_slices((X, y[..., tf.newaxis]))
        if shuffle:
            train_dataset = train_dataset.shuffle(y.shape[0])
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.prefetch(-1)

        if additional_datasets is not None:
            self.learning_curves = {}
            for name in ['train'] + list(additional_datasets.keys()):
                self.learning_curves[name + '_loss'] = []
                self.learning_curves[name + '_mae'] = []
                self.learning_curves[name + '_mre'] = []
                self.learning_curves[name + '_rmse'] = []
                self.learning_curves[name + '_r2'] = []

        for epoch in range(epochs):

            if verbose:
                train_dataset = tqdm(train_dataset)

            for batch in train_dataset:

                result = self._train_step(batch)

                if verbose:
                    current_lr = self.optimizer._decayed_lr(tf.float32).numpy()
                    description  = f'epoch {epoch:03d} : '
                    description += f'lr {current_lr:.6f} : '
                    description += f'loss {result["loss"]:5.3f} : '
                    dataset.set_description(description)

            if additional_datasets is not None:

                trues, preds = self.predict(X, y, verbose=0)
                self._accumulate_learning_curve(trues, preds, 'train')

                for name, dataset in additional_datasets.items():

                    trues, preds = self.predict(*dataset, verbose=0)
                    self._accumulate_learning_curve(trues, preds, name)

            if epoch >= (epochs * 0.8):
                decayed_lr = (self.initial_learning_rate * 0.1)
                self.optimizer.learning_rate.assign(decayed_lr)

        return self

    def predict(self, X, y, batch_size=32, verbose=0):

        dataset = tf.data.Dataset.from_tensor_slices((X, y[..., tf.newaxis]))
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(-1)

        if verbose:
            dataset = tqdm(dataset)

        preds, trues = [], []
        for batch in dataset:
            X, y = batch

            y_pred = self._predict_step(X)
            preds.append(np.squeeze(y_pred, axis=-1))
            trues.append(np.squeeze(y, axis=-1))

        return np.concatenate(trues, axis=0), np.concatenate(preds, axis=0)

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
