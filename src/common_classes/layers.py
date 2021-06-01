import tensorflow as tf


class GlobalNonZeroAveragePooling1D(tf.keras.layers.Layer):
    """Average pooling layer to deal with zero-padded graphs"""
    def call(self, x):
        # count the number of nonzero features, last axis
        nonzero = tf.math.reduce_any(tf.math.not_equal(x, 0.0), axis=-1)
        n = tf.reduce_sum(tf.cast(nonzero, 'float32'), axis=-1, keepdims=True)
        return tf.reduce_sum(x, axis=1) / n
