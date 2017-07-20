import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import argparse
import numpy as np
import src.matter as matter
import matplotlib.pyplot as plt
import cv2

mnist = input_data.read_data_sets("data/", one_hot=True)

parser = argparse.ArgumentParser()
parser.add_argument("--load", help="load weight", action="store_true")
parser.add_argument("--coeff", help="update rate", type=float)
parser.add_argument("--rate", help="learning rate", type=float)
parser.add_argument("--skip", help="example skip", type=int)
parser.add_argument("--limit", help="example limit", type=int)
parser.add_argument("--infer", help="total inference steps", type=int)
args = parser.parse_args()

if __name__ == '__main__':

    # print "Training set:", mnist.train.images.shape, mnist.train.labels.shape
    # print "Test set:", mnist.test.images.shape, mnist.test.labels.shape
    # print "Validation set:", mnist.validation.images.shape, mnist.validation.labels.shape

    data = np.concatenate([mnist.train.images, mnist.test.images, mnist.validation.images], axis=0)
    labels = np.concatenate([mnist.train.labels, mnist.test.labels, mnist.validation.labels], axis=0)

    layer_sizes = [400, 400, 200, 100]
    learning_coeff = 0.01 if not args.coeff else args.coeff
    learning_rate = 0.01 if not args.rate else args.rate
    run_skip = 0 if not args.skip else args.skip
    run_limit = 1000 if not args.limit else args.limit
    infer_steps = 100 if not args.infer else args.infer

    print "-----------------------"
    print "Dataset:", data.shape
    print "Labels:", labels.shape
    print "layer sizes: ", layer_sizes
    print "learning coeff: ", learning_coeff
    print "learning rate: ", learning_rate
    print "run skip: ", run_skip
    print "run limit: ", run_limit
    print "inference steps: ", infer_steps
    print "-----------------------"

    sess = tf.Session()
    model = matter.Autoencoder(sess, (data.shape[1], labels.shape[1]), layer_sizes, learning_coeff, learning_rate)
    sess.run(tf.global_variables_initializer())

    if args.load:
        print "loading..."
        model.load()

    error_graph = []
    average_error = 1.0
    for i in xrange(run_skip, run_skip + run_limit, 1):

        model.reset_labels(data[i:i + 1, :])

        projected_ = data[i:i + 1, :]
        for j in xrange(infer_steps):
            label_, projected_, _ = model.infer(data[i:i + 1, :])
            print _

        model.collect(data[i:i + 1, :], labels[i:i + 1, :])

        for j in xrange(infer_steps):
            model.learn(data[i:i + 1, :], labels[i:i + 1, :])

        average_error = learning_coeff * (np.sum((label_ - labels[i, :])**2)) + (1 - learning_coeff) * average_error
        print average_error
        error_graph.append(average_error)
        # model.debug_test()

        canvas = np.concatenate((np.reshape(data[i:i + 1, :], (28, 28)), np.reshape(projected_, (28, 28))), axis=1)
        cv2.imshow("a", canvas)
        cv2.waitKey(1)
    model.save()

    plt.plot(xrange(run_skip, run_skip + run_limit, 1), error_graph)
    plt.ylabel('average error')
    plt.show()
