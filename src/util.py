import numpy as np
import tensorflow as tf


def random_uniform(rows, cols):
    return (np.random.rand(rows, cols) - 0.5) * 0.001


def build_cpu_shift_mat(size):
    r12340 = np.arange(1, size + 1, 1, dtype=np.int32)
    r12340[size - 1] = 0
    cpu_shift = np.identity(size)[:, r12340]
    return cpu_shift


def cross_entropy(y, z, variables, rate=0.001, name="adam"):
    cost = tf.reduce_sum(tf.multiply(z, -tf.log(y)) + tf.multiply((1 - z), -tf.log(1 - y)))
    training_op = tf.train.AdamOptimizer(rate, name=name).minimize(cost, var_list=variables)
    return {"op": training_op, "cost": cost}


def l2_loss(y, z, variables, rate=0.001, name="adam"):
    cost = tf.reduce_sum(tf.squared_difference(z, y))
    training_op = tf.train.AdamOptimizer(rate, name=name).minimize(cost, var_list=variables)
    return {"op": training_op, "cost": cost}


def apply_gradients(gradients, delta, rate=0.001, name="adam"):
    training_op = tf.train.AdamOptimizer(rate, name=name).apply_gradients(gradients)
    if delta is not None:
        return {"op": training_op, "cost": delta}
    else:
        return {"op": training_op}


def apply_gradients_vanilla(gradients, delta, rate=0.001, name="gd"):
    training_op = tf.train.GradientDescentOptimizer(rate, name=name).apply_gradients(gradients)
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


# dist has shape [N, D]
# domain has shape [D, 2] containing min and max of each dimension.
def compute_basis_wise_entropy(dist, domain, bins=2):
    N = float(dist.shape[0])
    D = float(dist.shape[1])
    bin_sizes = (domain[:, 1] - domain[:, 0]) / bins
    indices = np.clip(((dist - domain[:, 0]) / bin_sizes).astype(np.int32), 0, bins - 1)

    counts = [np.sum(indices == i, axis=0) for i in range(bins)]
    unnorm = np.sum(np.sum(np.where(c > 0, c * np.log(c), 0)) for c in counts)

    return D * np.log(N) - unnorm / N


if __name__ == '__main__':
    data = np.random.uniform(size=(10, 2, 3))
    print data
    print prepare_data(data, -2, 3)
    print compute_basis_wise_entropy(
        np.random.rand(1024, 8),
        np.stack([np.zeros(8), np.ones(8)], axis=1))
