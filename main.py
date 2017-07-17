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

    layer_sizes = [data.shape[1] + labels.shape[1], 400, 200, 100, 100, 10]
    learning_coeff = 0.1
    run_skip = 0
    run_limit = 100

    print "-----------------------"
    print "Dataset:", data.shape
    print "Labels:", labels.shape
    print "layer sizes: ", layer_sizes
    print "learning coeff: ", learning_coeff
    print "run skip: ", run_skip
    print "run limit: ", run_limit
    print "-----------------------"

    sess = tf.Session()
    model = matter.Autoencoder(sess, layer_sizes, learning_coeff)
    sess.run(tf.global_variables_initializer())

    # if not args.reset:
    #     machine.load_session("./artifacts/demo")

    error_graph = []
    average_error = 1.0
    for i in xrange(run_skip, run_skip + run_limit, 1):
        label_ = model.infer(data[i, :])
        model.learn(data[i, :], labels[i, :])
        average_error = learning_coeff * (np.sum((label_ - labels[i, :])**2)) + (1 - learning_coeff) * average_error
        error_graph.append(average_error)

    plt.plot(xrange(run_skip, run_skip + run_limit, 1), error_graph)
    plt.ylabel('average error')
    plt.show()
