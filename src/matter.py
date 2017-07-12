import tensorflow as tf
import numpy as np
import util


class Matter:
    # implement overcoming catastrophic forgetting based on https://arxiv.org/pdf/1612.00796.pdf

    def __init__(self, layers, activation=tf.sigmoid, penalty=0.1, mov=0.999, stat_mov=0.99):
        self.f = activation
        self.mov = mov
        self.penalty = penalty
        self.stat_mov = stat_mov
        self.Ws = []
        self.Bs = []
        self.input_bias = tf.Variable(np.zeros((layers[0])), dtype=tf.float32)

        # each = (Fisher information, previous optimal parameters)
        self.vWs = []
        self.vBs = []
        self.vBias = [tf.Variable(np.zeros((layers[0])), dtype=tf.float32)] * 3

        self.unnormalized_partition = tf.Variable(1, dtype=tf.float32)

        for i in xrange(1, len(layers)):
            self.Ws.append(tf.Variable(util.random_uniform(layers[i - 1], layers[i]), dtype=tf.float32))
            self.vWs.append([tf.Variable(np.zeros((layers[i - 1], layers[i])), dtype=tf.float32)] * 3)
            self.Bs.append(tf.Variable(np.zeros((layers[i])), dtype=tf.float32))
            self.vBs.append([tf.Variable(np.zeros((layers[i])), dtype=tf.float32)] * 3)

    def debug(self):
        return [x[1] for x in self.vWs]

    def forward(self, input):
        output = input
        for i in xrange(0, len(self.Ws)):
            output = self.f(tf.matmul(output, self.Ws[i]) + self.Bs[i])
        return output

    def backward(self, output):
        input = output
        for i in xrange(len(self.Ws) - 1, 0, -1):
            input = self.f(tf.matmul(input, self.Ws[i], transpose_b=True) + self.Bs[i - 1])
        return self.f(tf.matmul(input, self.Ws[0], transpose_b=True) + self.input_bias)

    def gradients(self, cost):

        all_vars = self.Ws + self.Bs + [self.input_bias]
        all_Fs = self.vWs + self.vBs + [self.vBias]
        penalty_term = tf.constant(0, dtype=tf.float32)
        for i in xrange(len(all_vars)):
            penalty_term = penalty_term + tf.reduce_mean(tf.stop_gradient(all_Fs[i][0]) * tf.square(all_vars[i] - tf.stop_gradient(all_Fs[i][1])))
        objective = cost + penalty_term * self.penalty
        grads = tf.gradients(objective, all_vars)
        L_grads = tf.gradients(-cost, all_vars)
        likelihood = tf.exp(-cost)

        collect_stat_ops = []
        for i in xrange(len(L_grads)):
            if(L_grads[i] is None):
                continue
            collect_stat_ops.append(tf.assign(all_Fs[i][2], all_Fs[i][2] * self.stat_mov + likelihood * tf.square(L_grads[i]) * (1 - self.stat_mov)).op)
        collect_stat_ops.append(tf.assign(self.unnormalized_partition, self.unnormalized_partition * self.stat_mov + likelihood * (1 - self.stat_mov)).op)

        return zip(grads, all_vars), collect_stat_ops

    def get_reseed_operation(self):

        all_vars = self.Ws + self.Bs + [self.input_bias]
        all_Fs = self.vWs + self.vBs + [self.vBias]
        self.reseed_ops = []
        for i in xrange(len(all_vars)):
            self.reseed_ops.append(tf.assign(all_Fs[i][0], all_Fs[i][0] * self.mov + (1 - self.mov) * all_Fs[i][2] / self.unnormalized_partition).op)
            self.reseed_ops.append(tf.assign(all_Fs[i][1], all_vars[i]).op)

        return self.reseed_ops


if __name__ == '__main__':
    sess = tf.Session()

    input_size = 100
    matter = Matter([input_size, input_size, input_size], tf.sigmoid, penalty=1e4, mov=0.90, stat_mov=0.90)
    input = tf.placeholder(tf.float32, [1, input_size])
    output = tf.placeholder(tf.float32, [1, input_size])
    gen = matter.forward(input)
    cost = tf.reduce_sum(tf.squared_difference(gen, output))
    grads, stats = matter.gradients(cost)
    training_op = tf.train.AdamOptimizer(0.001).apply_gradients(grads)
    optimize_op = {"op": training_op, "cost": cost, "stat": stats}
    reseed = matter.get_reseed_operation()
    # grads = util.l2_loss(output, gen, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), 0.01)

    sess.run(tf.global_variables_initializer())

    inputs = []
    outputs = []
    for i in xrange(10):
        inp = np.random.rand(1, input_size)
        out = np.random.rand(1, input_size)
        inputs.append(inp)
        outputs.append(out)
        for j in xrange(500):
            print sess.run(optimize_op, feed_dict={input: inp, output: out})
        # print sess.run(matter.debug())
        print sess.run(reseed)

    for i in xrange(10):
        print sess.run(tf.reduce_sum(tf.squared_difference(output, gen)), feed_dict={input: inputs[i], output: outputs[i]})
