import tensorflow as tf


class GraphConvLayer(tf.keras.layers.Layer):

    def __init__(self,
                 units=128,
                 activation='relu',
                 dropout=0.0,
                 batch_norm=False,
                 initializer='glorot_uniform',
                 regularizer=None,
                 **kwargs):

        super(GraphConvLayer, self).__init__(**kwargs)

        self.units = units

        self.dropout = (
            tf.keras.layers.Dropout(dropout) if dropout
            else tf.keras.layers.Lambda(lambda x: x)
        )

        self.batch_norm = (
            tf.keras.layers.BatchNormalization() if batch_norm
            else tf.keras.layers.Lambda(lambda x: x)
        )

        self.activation = tf.keras.activations.get(activation)
        self.regularizer = tf.keras.regularizers.get(regularizer)
        self.initializer = tf.keras.initializers.get(initializer)

        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        return mask[1]

    def build(self, input_shape):

        self.W1 = self.add_weight(
            shape=(input_shape[1][-1], self.units),
            initializer=self.initializer,
            regularizer=self.regularizer,
            trainable=True,
            name='W1',
            dtype=tf.float32
        )
        self.W0 = self.add_weight(
            shape=(input_shape[1][-1], self.units),
            initializer=self.initializer,
            regularizer=self.regularizer,
            trainable=True,
            name='W0',
            dtype=tf.float32
        )

    def call(self, inputs, training=False, mask=None):
        A, H = inputs
        H1 = tf.matmul(A, H)
        H1 = tf.matmul(H1, self.W1)
        H0 = tf.matmul(H, self.W0)
        H = (H1 + H0)
        H = self.batch_norm(H, training=training)
        H = self.activation(H)
        H = self.dropout(H, training=training)
        if mask:
            H_mask = mask[1][:, :, None]
            H *= tf.cast(H_mask, H.dtype)
        return H
