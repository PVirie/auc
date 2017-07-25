import numpy as np
import tensorflow as tf


def random_uniform(rows, cols):
    return (np.random.rand(rows, cols) - 0.5) * 0.001


def build_cpu_shift_mat(size):
    r12340 = np.arange(1, size + 1, 1, dtype=np.int32)
    r12340[size - 1] = 0
    cpu_shift = np.identity(size)[:, r12340]
    return cpu_shift


def cross_entropy(y, z, variables, rate=0.001):
    cost = tf.reduce_sum(tf.multiply(z, -tf.log(y)) + tf.multiply((1 - z), -tf.log(1 - y)))
    training_op = tf.train.AdamOptimizer(rate).minimize(cost, var_list=variables)
    return {"op": training_op, "cost": cost}


def l2_loss(y, z, variables, rate=0.001):
    cost = tf.reduce_sum(tf.squared_difference(z, y))
    training_op = tf.train.AdamOptimizer(rate).minimize(cost, var_list=variables)
    return {"op": training_op, "cost": cost}


def apply_gradients(gradients, delta, rate=0.001):
    training_op = tf.train.AdamOptimizer(rate).apply_gradients(gradients)
    if delta is not None:
        return {"op": training_op, "cost": delta}
    else:
        return {"op": training_op}


def apply_gradients_vanilla(gradients, delta, rate=0.001):
    training_op = tf.train.GradientDescentOptimizer(rate).apply_gradients(gradients)
    if delta is not None:
        return {"op": training_op, "cost": delta}
    else:
        return {"op": training_op}


def tf_ones_or_zeros(c):
    ones = tf.ones(tf.shape(c), dtype=tf.float32)
    zeros = tf.zeros(tf.shape(c), dtype=tf.float32)
    return tf.where(c, ones, zeros)


def tf_random_binomial(p):
    return tf_ones_or_zeros(tf.random_uniform(tf.shape(p), 0, 1, dtype=tf.float32) < p)


def prepare_data(data, first, last_not_included):
    # data are of shape [len, ...]
    if first < 0:
        flat_size = np.prod(data.shape) / data.shape[0]
        temp = np.zeros((last_not_included - first, flat_size), dtype=np.float32)
        if last_not_included <= 0:
            return temp
        temp[(-first):(last_not_included - first), :] = np.reshape(data[0:last_not_included, ...], (last_not_included, flat_size))
        return temp
    else:
        return np.reshape(data[first:last_not_included, ...], (last_not_included - first, -1))


if __name__ == '__main__':
    data = np.random.uniform(size=(10, 2, 3))
    print data
    print prepare_data(data, -2, 3)
