import tensorflow as tf
import numpy as np
from abc import ABCMeta, abstractmethod
from typing import List


class BaseDataset(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, filenames, batch_size, training, num_parallel_calls):
        self.filenames = filenames
        self.batch_size = batch_size
        self.training = training
        self.num_parallel_calls = num_parallel_calls
        self.random_seed = 42

    def __len__(self):
        # slow way of obtaining the number of examples of the dataset
        dataset = tf.data.TFRecordDataset(
            filenames=self.filenames,
            num_parallel_reads=self.num_parallel_calls).repeat(1)
        return dataset.reduce(np.int64(0), lambda x, _: x + 1)

    @property
    @abstractmethod
    def padded_shapes(self):
        return {'placeholder': (None,)}

    @abstractmethod
    def preprocess_function(self, features):
        return {'placeholder': tf.constant([0.0])}

    def _parse_function(self, example_proto):
        features = [
            ('index', tf.int64),
            ('label', tf.float32),
            ('string', tf.string),
            ('feature_indices', tf.int64),
            ('feature_values', tf.float32),
            ('feature_shape', tf.int64),
            ('feature_min', tf.float32),
            ('feature_max', tf.float32),
            ('adjacency_indices', tf.int64),
            ('adjacency_values', tf.float32),
            ('adjacency_shape', tf.int64),
            ('edge_indices', tf.int64),
            ('edge_values', tf.float32),
            ('edge_shape', tf.int64),
            ('edge_min', tf.float32),
            ('edge_max', tf.float32),
        ]
        feature_descriptions = {
            f[0]: tf.io.FixedLenFeature([], tf.string)
            for f in features
        }
        example = tf.io.parse_single_example(example_proto, feature_descriptions)
        features = {
            f[0]: tf.io.parse_tensor(example[f[0]], out_type=f[1])
            for f in features
        }
        return features

    def _get_parsed_tfrecord_dataset(self) -> tf.data.Dataset:
        dataset = tf.data.TFRecordDataset(
            filenames=self.filenames,
            num_parallel_reads=self.num_parallel_calls).repeat(1)
        dataset = dataset.map(self._parse_function, self.num_parallel_calls)
        if self.training:
            dataset = dataset.shuffle(len(self), seed=self.random_seed)
        return dataset

    def get_labels(self) -> np.ndarray:
        dataset = self._get_parsed_tfrecord_dataset()
        dataset = dataset.map(lambda x: x['label'], self.num_parallel_calls)
        return np.array([label.numpy() for label in dataset])

    def get_strings(self) -> np.ndarray:
        dataset = self._get_parsed_tfrecord_dataset()
        dataset = dataset.map(lambda x: x['string'], self.num_parallel_calls)
        return np.array([string.numpy().decode('utf-8') for string in dataset])

    def get_indices(self) -> np.ndarray:
        # slow way of obtaining the indices of the examples in the dataset
        dataset = self._get_parsed_tfrecord_dataset()
        dataset = dataset.map(lambda x: x['index'], self.num_parallel_calls)
        return np.array([index.numpy() for index in dataset])

    def get_nonzero_features(self) -> np.ndarray:
        dataset = self._get_parsed_tfrecord_dataset()
        for x in dataset.take(1):
            keep_idx = tf.squeeze(
                tf.where(x['feature_min']-x['feature_max'] != 0))
        return keep_idx

    def get_iterator(self, filter_indices=None) -> tf.data.Dataset:
        dataset = self._get_parsed_tfrecord_dataset()
        if filter_indices is not None:
            dataset = dataset.filter(
                lambda x: tf.math.reduce_any(
                    tf.math.equal(filter_indices, x['index'])
                )
            )
        dataset = dataset.map(self.preprocess_function, self.num_parallel_calls)
        dataset = dataset.padded_batch(self.batch_size, padded_shapes=self.padded_shapes)
        return dataset.prefetch(self.num_parallel_calls)
