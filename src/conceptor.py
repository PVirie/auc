import tensorflow as tf
import numpy as np
import util

# implementing Xu He, Herbert Jaeger: Overcoming Catastrophic Interference by Conceptors


class Layer:

    def __init__(self, shape, aperture=1.0):
        self.shape = shape
        self.aperture = aperture
        self.w_ = tf.Variable(np.zeros((shape[0], shape[1])), dtype=tf.float32, name="w_")
        self.w = tf.Variable(np.zeros((shape[0], shape[1])), dtype=tf.float32, name="w")
        self.V = tf.Variable(np.zeros((shape[0], shape[0])), dtype=tf.float32, name="V")

    def create_graph(self, input, learning_coeff):

        init_w_ = tf.assign(self.w_, tf.random_uniform((self.shape[0], self.shape[1]), -1, 1) * 0.01).op

        F = tf.eye(self.shape[0]) - tf.matmul(self.V, tf.matrix_inverse(self.V + self.aperture * tf.eye(self.shape[0])))
        w_plus = self.w + tf.matmul(F, self.w_)
        output = tf.nn.elu(tf.matmul(input, w_plus))

        update_w = tf.assign(self.w, w_plus).op

        gather_V = tf.assign(self.V, tf.matmul(input, input, transpose_a=True) * learning_coeff + self.V * (1 - learning_coeff)).op
        clear_w_ = tf.assign(self.w_, tf.zeros((self.shape[0], self.shape[1]))).op

        return output, init_w_, update_w, (clear_w_, gather_V)

    def get_optimizable_variables(self):
        return [self.w_]


class MLP:

    def __init__(self, sess, input_size, layer_sizes, learning_rate=0.01, aperture=1.0):
        self.sess = sess
        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.layers = []

        s0_ = input_size
        for s1_ in layer_sizes:
            self.layers.append(Layer((s0_, s1_), aperture))
            s0_ = s1_

        self.learning_coeff = tf.placeholder(tf.float32, ())
        self.gpu_inputs = tf.placeholder(tf.float32, [1, input_size])
        self.gpu_labels = tf.placeholder(tf.float32, [1, s0_])

        output_, self.init_ops, self.update_ops, self.gather_ops, scope = self._create_forward_graph(self.gpu_inputs, self.learning_coeff)
        self.output = tf.argmax(output_, 1)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.gpu_labels, logits=output_))

        print [s.name for s in scope]
        self.learn_ops = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, var_list=scope)
        self.saver = tf.train.Saver(var_list=scope, keep_checkpoint_every_n_hours=1)

    def _create_forward_graph(self, input, learning_coeff):
        init_ops = []
        update_ops = []
        gather_ops = []
        variables = []
        v = input
        for l in self.layers:
            v, init_op, update_op, gather_op = l.create_graph(v, learning_coeff)
            init_ops.append(init_op)
            update_ops.append(update_op)
            gather_ops.append(gather_op)
            variables = variables + l.get_optimizable_variables()
        return v, init_ops, update_ops, gather_ops, variables

    def init(self):
        self.sess.run(self.init_ops)

    def learn(self, data, label):
        return self.sess.run((self.learn_ops, self.loss), feed_dict={self.gpu_inputs: data, self.gpu_labels: label})

    def update(self):
        self.sess.run(self.update_ops)

    def collect(self, data, significance):
        self.sess.run(self.gather_ops, feed_dict={self.gpu_inputs: data, self.learning_coeff: significance})

    def gen(self, data):
        return self.sess.run(self.output, feed_dict={self.gpu_inputs: data})

    def debug_test(self):
        print "stub"

    def save(self):
        self.saver.save(self.sess, "./artifacts/" + "weights")

    def load(self):
        self.saver.restore(self.sess, "./artifacts/" + "weights")


if __name__ == '__main__':

    sess = tf.Session()

    input = tf.placeholder(tf.float32, [None, None])
    output = tf.placeholder(tf.float32, [None, None])
    layer = Layer([10, 5])
    y, init_ops, update_ops, clear_ops = layer.create_graph(input, 0.001)
    error = tf.reduce_mean(tf.squared_difference(y, output))

    train_op = tf.train.AdamOptimizer(0.01).minimize(error, var_list=layer.get_optimizable_variables())

    sess.run(tf.global_variables_initializer())

    cpu_input = (np.random.rand(100, 10) - 0.5)
    cpu_output = np.matmul(cpu_input, np.random.rand(10, 5))

    results = []
    for i in xrange(cpu_input.shape[0]):
        sess.run(init_ops)
        for j in xrange(100):
            sess.run(train_op, feed_dict={input: cpu_input[i: i + 1, ...], output: cpu_output[i: i + 1, ...]})

        sess.run(update_ops)
        sess.run(clear_ops, feed_dict={input: cpu_input[i: i + 1, ...]})

        results.append(sess.run(error, feed_dict={input: cpu_input[i: i + 1, ...], output: cpu_output[i: i + 1, ...]}))

    for i in xrange(cpu_input.shape[0]):
        print results[i], sess.run(error, feed_dict={input: cpu_input[i: i + 1, ...], output: cpu_output[i: i + 1, ...]})
