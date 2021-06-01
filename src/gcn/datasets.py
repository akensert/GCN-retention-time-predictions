import tensorflow as tf

from base_classes import datasets


class GCNDataset(datasets.BaseDataset):

    def __init__(self,
                 filenames: str = '../input/tfrecords/train.tfrec',
                 batch_size: int = 128,
                 training: bool = False,
                 num_parallel_calls: int = tf.data.experimental.AUTOTUNE) -> None:
        super().__init__(
            filenames=filenames,
            batch_size=batch_size,
            training=training,
            num_parallel_calls=num_parallel_calls)

    @property
    def padded_shapes(self):
        return {
            'label': (None,),
            'adjacency_matrix': (None, None),
            'feature_matrix': (None, None),
            'string': (None,),
            'index': (None,),
        }

    def preprocess_function(self, features: dict) -> dict:

        adjacency_matrix = tf.scatter_nd(
            indices=features['adjacency_indices'],
            updates=features['adjacency_values'],
            shape=features['adjacency_shape']
        )

        # obtain sqrt(inv(degree)) to normalize adjacency matrix
        degree_matrix = tf.reduce_sum(adjacency_matrix, axis=-1)
        degree_matrix = tf.where(degree_matrix == 0.0, 1.0, degree_matrix)
        degree_matrix = tf.linalg.diag(degree_matrix)
        degree_matrix = tf.linalg.sqrtm(tf.linalg.inv(degree_matrix))

        adjacency_matrix = tf.matmul(
            tf.matmul(degree_matrix, adjacency_matrix), degree_matrix)

        feature_matrix = tf.scatter_nd(
            indices=features['feature_indices'],
            updates=features['feature_values'],
            shape=features['feature_shape']
        )

        # Remove empty features and rescale all features to be in range 0 to 1
        keep_idx = tf.squeeze(
            tf.where(features['feature_min']-features['feature_max'] != 0))
        feature_matrix = tf.gather(feature_matrix, keep_idx, axis=-1)

        fmin = tf.gather(features['feature_min'], keep_idx)
        fmax = tf.gather(features['feature_max'], keep_idx)
        feature_matrix = (feature_matrix-fmin)/(fmax - fmin)

        # if tf.reduce_sum(adjacency_matrix) == 0:
        #     adjacency_matrix = tf.constant([[1.]])

        return {
            'label': [features['label']],
            'adjacency_matrix': adjacency_matrix,
            'feature_matrix': feature_matrix,
            'string': [features['string']],
            'index': [features['index']],
        }
