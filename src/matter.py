import tensorflow as tf
import numpy as np
import util


def mirroring_relus(input):
    return tf.stack([tf.nn.relu(input), -tf.nn.relu(-input)], axis=2)


class Layer:
    def __init__(self, shape, learning_coeff, learning_rate=0.01):
        self.shape = shape
        self.learning_coeff = learning_coeff
        self.learning_rate = learning_rate
        self.w = tf.Variable(util.random_uniform(shape[0] * 2, shape[1]), dtype=tf.float32)
        self.b = tf.Variable(np.zeros((shape[0] * 2)), dtype=tf.float32)

        self.C = tf.Variable(np.zeros((shape[0] * 2, shape[0] * 2)), dtype=tf.float32)
        self.B = tf.Variable(np.zeros((shape[0] * 2)), dtype=tf.float32)

    def create_forward_graph(self, input):
        expanded = tf.reshape(mirroring_relus(input), [-1, 2 * self.shape[0]])
        biased = expanded - self.b
        reduced = tf.matmul(biased, self.w)

        gather_B = tf.assign(self.B, tf.reduce_mean(expanded, axis=0) * (self.learning_coeff) + self.B * (1 - self.learning_coeff)).op
        gather_C = tf.assign(self.C, tf.matmul(biased, biased, transpose_a=True) * (self.learning_coeff) + self.C * (1 - self.learning_coeff)).op

        learn_op_b = tf.assign(self.b, self.B).op

        wwt = tf.matmul(self.w, self.w, transpose_b=True)
        objective = tf.trace(self.C - tf.matmul(self.C, wwt) - tf.matmul(wwt, self.C) + tf.matmul(tf.matmul(wwt, self.C), wwt))
        grad_C = tf.gradients(objective, [self.w])
        learn_op_w = util.apply_gradients(zip(grad_C, [self.w]), objective, self.learning_rate)

        return reduced, (gather_C, gather_B, learn_op_b), learn_op_w

    def create_backward_graph(self, input):
        inverted_reduced = tf.matmul(input, self.w, transpose_b=True)
        inverted_biased = inverted_reduced + self.b
        inverted_expanded = tf.reduce_sum(tf.reshape(inverted_biased, [-1, self.shape[0], 2]), axis=2)
        return inverted_expanded

    def debug(self):
        return tf.reduce_mean(self.w)


class Autoencoder:

    def __init__(self, sess, input_size, layer_sizes, learning_coeff, learning_rate=0.01):
        self.sess = sess
        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.learning_coeff = learning_coeff
        self.layers = []
        self.debug = []

        s0_ = input_size[0] + input_size[1]
        for s1_ in layer_sizes:
            self.layers.append(Layer((s0_, s1_), learning_coeff, learning_rate))
            s0_ = s1_
            self.debug.append(self.layers[len(self.layers) - 1].debug())

        self.gpu_inputs = tf.placeholder(tf.float32, [1, input_size[0]])
        self.gpu_labels = tf.placeholder(tf.float32, [1, input_size[1]])

        _, collect_ops, learn_ops = self._create_forward_graph(self.gpu_inputs, self.gpu_labels)
        self.collect_ops = collect_ops
        self.learn_ops = learn_ops

        self.info_space = tf.Variable(np.zeros((1, s0_)), dtype=tf.float32)
        self.info_space_start, _, __ = self._create_forward_graph(self.gpu_inputs, self.gpu_labels)
        self.reset_ops = tf.assign(self.info_space, self.info_space_start).op

        self.projected_input, self.projected_label = self._create_backward_graph(self.info_space)
        infer_objective = tf.reduce_sum(tf.squared_difference(self.gpu_inputs, self.projected_input))
        infer_grad = tf.gradients(infer_objective, [self.info_space])
        self.infer_ops = util.apply_gradients(zip(infer_grad, [self.info_space]), infer_objective, learning_rate)

        scope = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.saver = tf.train.Saver(var_list=scope, keep_checkpoint_every_n_hours=1)

    def _create_forward_graph(self, input, label):
        collect_ops = []
        learn_ops = []
        v = tf.concat([input, label], axis=1)
        for l in self.layers:
            v, collect_op, learn_op = l.create_forward_graph(v)
            collect_ops.append(collect_op)
            learn_ops.append(learn_op)
        return v, collect_ops, learn_ops

    def _create_backward_graph(self, input):
        h = input
        for l in reversed(self.layers):
            h = l.create_backward_graph(h)
        return tf.slice(h, [0, 0], [-1, self.input_size[0]]), tf.slice(h, [0, self.input_size[0]], [-1, self.input_size[1]])

    def collect(self, data, label):
        self.sess.run(self.collect_ops, feed_dict={self.gpu_inputs: data, self.gpu_labels: label})

    def learn(self, data, label):
        self.sess.run(self.learn_ops, feed_dict={self.gpu_inputs: data, self.gpu_labels: label})

    def reset_labels(self, data):
        self.sess.run(self.reset_ops, feed_dict={self.gpu_inputs: data, self.gpu_labels: np.zeros((1, self.input_size[1]))})

    def infer(self, data):
        return self.sess.run((self.projected_label, self.projected_input, self.infer_ops), feed_dict={self.gpu_inputs: data})

    def debug_test(self):
        print self.sess.run(self.debug)

    def save(self):
        self.saver.save(self.sess, "./artifacts/" + "weights")

    def load(self):
        self.saver.restore(self.sess, "./artifacts/" + "weights")


if __name__ == '__main__':

    sess = tf.Session()

    input = tf.placeholder(tf.float32, [None, None])
    output = mirroring_relus(input)
    input_ = tf.reduce_sum(output, 2)
    error = tf.reduce_sum(tf.squared_difference(input_, input))
    cpu_input = (np.random.rand(1, 10) - 0.5)
    print sess.run(error, feed_dict={input: cpu_input})
