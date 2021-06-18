import tensorflow as tf

from base_classes import models
from . import layers


class RGCNModel(models.BaseModel):

    def __init__(self,
                 loss_fn=tf.keras.losses.Huber,
                 optimizer=tf.keras.optimizers.Adam,
                 initial_learning_rate=1e-3,
                 gconv_num_bases=-1,
                 gconv_units=[128, 128, 128, 128],
                 gconv_activation='relu',
                 gconv_dropout=0.0,
                 gconv_batch_norm=False,
                 gconv_initializer='glorot_uniform',
                 gconv_regularizer=None,
                 dense_units=[1024,],
                 dense_activation='relu',
                 dense_dropout=0.0,
                 dense_initializer='glorot_uniform',
                 dense_regularizer=None,
                 **kwargs):

        super().__init__(
            loss_fn=loss_fn,
            optimizer=optimizer,
            initial_learning_rate=initial_learning_rate,
            end_learning_rate=None,  # NOT USED
            power=None,  # NOT USED
            **kwargs)

        self.masking = tf.keras.layers.Masking(mask_value=0)

        self.gconv_layers = [
            layers.RelationalGraphConvLayer(
                units=units,
                num_bases=gconv_num_bases,
                activation=gconv_activation,
                dropout=gconv_dropout,
                batch_norm=gconv_batch_norm,
                initializer=gconv_initializer,
                regularizer=gconv_regularizer,
            )
            for units in gconv_units
        ]

        self.pooling = tf.keras.layers.GlobalAveragePooling1D()

        self.dense_layers = [
            tf.keras.layers.Dense(
                units=units,
                activation=dense_activation,
                kernel_initializer=dense_initializer,
                kernel_regularizer=dense_regularizer
            )
            for units in dense_units
        ]

        self.dense_dropout = tf.keras.layers.Dropout(dense_dropout)

        self.dense_output = tf.keras.layers.Dense(1,  dtype='float32')

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training):

        A, H = inputs
        H = self.masking(H)
        for i in range(len(self.gconv_layers)):
            H = self.gconv_layers[i]([A, H], training=training)

        Z = self.pooling(H)

        for i in range(len(self.dense_layers)):
            Z = self.dense_layers[i](Z)
            Z = self.dense_dropout(Z, training=training)

        return self.dense_output(Z)
