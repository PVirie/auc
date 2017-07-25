import tensorflow as tf
import numpy as np
import util


def mirroring_relus(input):
    return tf.stack([tf.nn.relu(input), tf.nn.relu(-input)], axis=2)


def inverse_mirroring_relus(input):
    bases = tf.unstack(input, axis=2)
    conditions = bases[0] > bases[1]
    out = tf.where(conditions, bases[0], -bases[1])
    residue = tf.where(conditions, tf.square(bases[1]), tf.square(bases[0]))
    return out, residue


class Layer:
    def __init__(self, shape, learning_rate=0.01, moving_average=True):
        self.shape = shape
        self.learning_rate = learning_rate
        self.moving_average = moving_average
        self.w = tf.Variable(util.random_uniform(shape[0] * 2, shape[1]), dtype=tf.float32)
        self.b = tf.Variable(np.zeros((shape[0] * 2)), dtype=tf.float32)

        self.C = tf.Variable(np.zeros((shape[0] * 2, shape[0] * 2)), dtype=tf.float32)
        self.B = tf.Variable(np.zeros((shape[0] * 2)), dtype=tf.float32)

    def create_forward_graph(self, input, learning_coeff, num):
        expanded = tf.reshape(mirroring_relus(input), [-1, 2 * self.shape[0]])
        biased = expanded - self.b
        reduced = tf.matmul(biased, self.w)

        if self.moving_average:
            gather_B = tf.assign(self.B, tf.reduce_mean(expanded, axis=0) * (learning_coeff) + self.B * (1 - learning_coeff)).op
            gather_C = tf.assign(self.C, tf.matmul(biased, biased, transpose_a=True) * (learning_coeff) + self.C * (1 - learning_coeff)).op
        else:
            gather_B = tf.assign(self.B, (tf.reduce_mean(expanded, axis=0) + self.B * (num - 1)) / num).op
            gather_C = tf.assign(self.C, (tf.matmul(biased, biased, transpose_a=True) + self.C * (num - 1)) / num).op

        learn_op_b = tf.assign(self.b, self.B).op

        wwt = tf.matmul(self.w, self.w, transpose_b=True)
        objective = tf.trace(self.C - tf.matmul(self.C, wwt) - tf.matmul(wwt, self.C) + tf.matmul(tf.matmul(wwt, self.C), wwt))
        grad_C = tf.gradients(objective, [self.w])
        learn_op_w = util.apply_gradients(zip(grad_C, [self.w]), objective, self.learning_rate)

        return reduced, (gather_C, gather_B, learn_op_b), learn_op_w

    def create_backward_graph(self, input):
        inverted_reduced = tf.matmul(input, self.w, transpose_b=True)
        inverted_biased = inverted_reduced + self.b
        inverted_expanded, residue = inverse_mirroring_relus(tf.reshape(inverted_biased, [-1, self.shape[0], 2]))
        return inverted_expanded, residue

    def debug(self):
        return tf.reduce_mean(self.w)


class Autoencoder:

    def __init__(self, sess, input_size, layer_sizes, learning_rate=0.01, moving_average=True):
        self.sess = sess
        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.layers = []
        self.debug = []

        s0_ = input_size[0] + input_size[1]
        for s1_ in layer_sizes:
            self.layers.append(Layer((s0_, s1_), learning_rate, moving_average))
            s0_ = s1_
            self.debug.append(self.layers[len(self.layers) - 1].debug())

        self.learning_coeff = tf.placeholder(tf.float32, ())
        self.gpu_inputs = tf.placeholder(tf.float32, [1, input_size[0]])
        self.gpu_labels = tf.placeholder(tf.float32, [1, input_size[1]])

        self.num_examples = tf.Variable(0, dtype=tf.int64)
        self.input_space = tf.Variable(np.zeros((1, input_size[0])), dtype=tf.float32)
        self.info_space = tf.Variable(np.zeros((1, s0_)), dtype=tf.float32)

        self.info_space_start, collect_ops, learn_ops = self._create_forward_graph(
            self.gpu_inputs, self.gpu_labels,
            self.learning_coeff, tf.to_float(tf.assign_add(self.num_examples, 1)))

        self.reset_ops = tf.assign(self.input_space, self.gpu_inputs).op
        self.project_ops = tf.assign(self.info_space, self.info_space_start).op
        self.collect_ops = collect_ops
        self.learn_ops = learn_ops

        self.projected_input, self.projected_label, self.residues = self._create_backward_graph(self.info_space)
        infer_objective = tf.reduce_sum(tf.squared_difference(self.input_space, self.projected_input)) + self.residues
        infer_grad = tf.gradients(infer_objective, [self.info_space])

        with tf.variable_scope("infer_optimizer"):
            self.infer_ops = util.apply_gradients(zip(infer_grad, [self.info_space]), infer_objective, learning_rate)

        infer_optimizer_scope = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="infer_optimizer")
        self.reset_infer_ops = tf.variables_initializer(infer_optimizer_scope)

        scope = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.saver = tf.train.Saver(var_list=scope, keep_checkpoint_every_n_hours=1)

    def _create_forward_graph(self, input, label, learning_coeff, num):
        collect_ops = []
        learn_ops = []
        v = tf.concat([input, label], axis=1)
        for l in self.layers:
            v, collect_op, learn_op = l.create_forward_graph(v, learning_coeff, num)
            collect_ops.append(collect_op)
            learn_ops.append(learn_op)
        return v, collect_ops, learn_ops

    def _create_backward_graph(self, input):
        h = input
        residues = tf.zeros((), dtype=tf.float32)
        for l in reversed(self.layers):
            h, r = l.create_backward_graph(h)
            residues = residues + tf.reduce_sum(r)
        return tf.slice(h, [0, 0], [-1, self.input_size[0]]), tf.slice(h, [0, self.input_size[0]], [-1, self.input_size[1]]), residues

    def collect(self, data, label, significance):
        self.sess.run(self.collect_ops, feed_dict={self.gpu_inputs: data, self.gpu_labels: label, self.learning_coeff: significance})

    def learn(self):
        return self.sess.run(self.learn_ops)

    def reset_input(self, data, label):
        self.sess.run((self.reset_ops, self.reset_infer_ops), feed_dict={self.gpu_inputs: data, self.gpu_labels: label})

    def reset_info(self, data, label):
        self.sess.run(self.project_ops, feed_dict={self.gpu_inputs: data, self.gpu_labels: label})

    def infer(self):
        return self.sess.run((self.projected_label, self.projected_input, self.infer_ops))

    def debug_test(self):
        print self.sess.run(self.num_examples)

    def save(self):
        self.saver.save(self.sess, "./artifacts/" + "weights")

    def load(self):
        self.saver.restore(self.sess, "./artifacts/" + "weights")


if __name__ == '__main__':

    sess = tf.Session()

    input = tf.placeholder(tf.float32, [None, None])
    output = mirroring_relus(input)
    input_, residue = inverse_mirroring_relus(output)
    error = tf.reduce_sum(tf.squared_difference(input_, input))
    cpu_input = (np.random.rand(1, 10) - 0.5)
    print sess.run((error, input_, input), feed_dict={input: cpu_input})
