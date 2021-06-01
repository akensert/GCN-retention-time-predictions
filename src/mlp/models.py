import tensorflow as tf
import numpy as np
from tqdm import tqdm


class ANNModel(tf.keras.Model):

    def __init__(self,
                 hidden_units=[256,],
                 kernel_regularizer=None,
                 dropout_rate=0.0,
                 initial_learning_rate=1e-3,
                 end_learning_rate=5e-5,
                 power=2.0,
                 loss_fn=tf.keras.losses.Huber,
                 optimizer=tf.keras.optimizers.Adam,
                 **kwargs):

        super().__init__(**kwargs)

        self.initial_learning_rate = initial_learning_rate
        self.end_learning_rate = end_learning_rate
        self.power = power

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

    @staticmethod
    def decayed_learning_rate(current_epoch,
                              initial_learning_rate,
                              end_learning_rate,
                              total_epochs,
                              power):
        current_epoch = min(current_epoch, total_epochs)
        decayed_lr = (
            (initial_learning_rate - end_learning_rate) *
            (1 - current_epoch / total_epochs)**power
        ) + end_learning_rate
        return decayed_lr

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

    def fit(self, X, y, batch_size=32, epochs=1, steps_per_epoch=None, shuffle=True, verbose=0):

        dataset = tf.data.Dataset.from_tensor_slices((X, y[..., tf.newaxis]))
        if shuffle:
            dataset = dataset.shuffle(y.shape[0])
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(-1)

        for epoch in range(epochs):

            if verbose:
                dataset = tqdm(dataset)

            for batch in dataset:

                result = self._train_step(batch)

                if verbose:
                    current_lr = self.optimizer._decayed_lr(tf.float32).numpy()
                    description  = f'epoch {epoch:03d} : '
                    description += f'lr {current_lr:.6f} : '
                    description += f'loss {result["loss"]:5.3f} : '
                    dataset.set_description(description)

            decayed_lr = self.decayed_learning_rate(
                epoch, self.initial_learning_rate, self.end_learning_rate,
                epochs, self.power
            )
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
