import tensorflow as tf
import sonnet as snt


class Normalizer(snt.Module):
    def __init__(self, size, max_accumulations=10**6, std_epsilon=1e-8, name='Normalizer'):
        super(Normalizer, self).__init__(name=name)
        self._max_accumulations = max_accumulations
        self._std_epsilon = std_epsilon

        self._acc_count = tf.Variable(0, dtype=tf.float32, name='acc_count', trainable=False)
        self._num_accumulations = tf.Variable(0, dtype=tf.float32, name='num_accumulations', trainable=False)
        self._acc_sum = tf.Variable(tf.zeros(size, tf.float32), name='acc_sum', trainable=False)
        self._acc_sum_squared = tf.Variable(tf.zeros(size, tf.float32), name='acc_sum_squared', trainable=False)


    def __call__(self, batched_data, accumulate=True):
        update_op = tf.no_op()
        if accumulate:
            update_op = tf.cond(self._num_accumulations < self._max_accumulations, 
                                lambda: self._accumulate(batched_data), 
                                tf.no_op)
        with tf.control_dependencies([update_op]):
            return (batched_data - self._mean()) / self._std_with_epsilon()


    def inverse(self, normalized_batch_data):
        return normalized_batch_data * self._std_with_epsilon() + self._mean()


    def _accumulate(self, batched_data):
        count = tf.cast(tf.shape(batched_data)[0], tf.float32)
        data_sum = tf.reduce_sum(batched_data, axis=0)
        squared_data_sum = tf.reduce_sum(batched_data**2, axis=0)

        return tf.group(
            self._acc_sum.assign_add(data_sum),
            self._acc_sum_squared.assign_add(squared_data_sum),
            self._acc_count.assign_add(count),
            self._num_accumulations.assign_add(1.)
        )


    def _mean(self):
        safe_count = tf.maximum(self._acc_count, 1.)
        return self._acc_sum / safe_count


    def _std_with_epsilon(self):
        safe_count = tf.maximum(self._acc_count, 1.)
        std = tf.sqrt(self._acc_sum_squared / safe_count - self._mean() ** 2)
        return tf.math.maximum(std, self._std_epsilon)

