import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import argparse
import numpy as np
import src.matter as matter
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("data/", one_hot=True)

parser = argparse.ArgumentParser()
parser.add_argument("--reset", help="reset weight", action="store_true")
parser.add_argument("--rate", help="learning rate", type=float)
parser.add_argument("--total", help="total maps", type=int)
parser.add_argument("--gen", help="generate mode, ignore total flag", action="store_true")
args = parser.parse_args()

if __name__ == '__main__':

    # print "Training set:", mnist.train.images.shape, mnist.train.labels.shape
    # print "Test set:", mnist.test.images.shape, mnist.test.labels.shape
    # print "Validation set:", mnist.validation.images.shape, mnist.validation.labels.shape

    data = np.concatenate([mnist.train.images, mnist.test.images, mnist.validation.images], axis=0)
    labels = np.concatenate([mnist.train.labels, mnist.test.labels, mnist.validation.labels], axis=0)

    layer_sizes = [400, 200, 100, 10]
    learning_coeff = 0.01
    run_skip = 0
    run_limit = 1000

    print "-----------------------"
    print "Dataset:", data.shape
    print "Labels:", labels.shape
    print "layer sizes: ", layer_sizes
    print "learning coeff: ", learning_coeff
    print "run skip: ", run_skip
    print "run limit: ", run_limit
    print "-----------------------"

    sess = tf.Session()
    model = matter.Autoencoder(sess, (data.shape[1], labels.shape[1]), layer_sizes, learning_coeff)
    sess.run(tf.global_variables_initializer())

    error_graph = []
    average_error = 1.0
    for i in xrange(run_skip, run_skip + run_limit, 1):

        projected_ = data[i:i + 1, :]
        label_ = np.zeros((1, labels.shape[1]))
        for j in xrange(100):
            projected_, label_ = model.infer(projected_, label_)

        model.learn(data[i:i + 1, :], labels[i:i + 1, :])
        print label_
        average_error = learning_coeff * (np.sum((label_ - labels[i, :])**2)) + (1 - learning_coeff) * average_error
        error_graph.append(average_error)
        # model.debug_test()

    plt.plot(xrange(run_skip, run_skip + run_limit, 1), error_graph)
    plt.ylabel('average error')
    plt.show()