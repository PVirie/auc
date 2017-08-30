import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import argparse
import numpy as np
import src.conceptor as conceptor
import matplotlib.pyplot as plt
import cv2
import random

mnist = input_data.read_data_sets("data/", one_hot=True)

parser = argparse.ArgumentParser()
parser.add_argument("--layers", help="ex: '100, 100, 100'")
parser.add_argument("--load", help="load weight", action="store_true")
parser.add_argument("--coeff", help="update rate", type=float)
parser.add_argument("--eval", help="evaluation coefficient", type=float)
parser.add_argument("--rate", help="learning rate", type=float)
parser.add_argument("--skip", help="run skip", type=int)
parser.add_argument("--limit", help="run limit", type=int)
parser.add_argument("--boot_skip", help="bootstrap skip", type=int)
parser.add_argument("--boot_limit", help="bootstrap limit", type=int)
parser.add_argument("--infer", help="total inference steps", type=int)
args = parser.parse_args()

if __name__ == '__main__':

    # print "Training set:", mnist.train.images.shape, mnist.train.labels.shape
    # print "Test set:", mnist.test.images.shape, mnist.test.labels.shape
    # print "Validation set:", mnist.validation.images.shape, mnist.validation.labels.shape

    data = np.concatenate([mnist.train.images, mnist.test.images, mnist.validation.images], axis=0)
    labels = np.concatenate([mnist.train.labels, mnist.test.labels, mnist.validation.labels], axis=0)

    layer_sizes = [64, 64, 10] if not args.layers else [int(x) for x in args.layers.split(',')]
    learning_coeff = 0.001 if not args.coeff else args.coeff
    eval_coeff = 0.01 if not args.eval else args.eval
    learning_rate = 0.001 if not args.rate else args.rate
    run_skip = 0 if not args.skip else args.skip
    run_limit = 1000 if not args.limit else args.limit
    bootstrap_skip = run_skip if not args.boot_skip else args.boot_skip
    bootstrap_limit = 1 if not args.boot_limit else args.boot_limit
    infer_steps = 100 if not args.infer else args.infer

    print "-----------------------"
    print "Dataset:", data.shape
    print "Labels:", labels.shape
    print "layer sizes: ", layer_sizes
    print "learning coeff: ", learning_coeff
    print "evaluation coeff: ", eval_coeff
    print "learning rate: ", learning_rate
    print "run skip: ", run_skip
    print "run limit: ", run_limit
    print "bootstrap skip: ", bootstrap_skip
    print "bootstrap limit: ", bootstrap_limit
    print "inference steps: ", infer_steps
    print "-----------------------"

    sess = tf.Session()
    model = conceptor.MLP(sess, data.shape[1], layer_sizes, learning_rate, aperture=1.0)
    sess.run(tf.global_variables_initializer())

    if args.load:
        print "loading..."
        model.load()

    error_graph = []
    average_error = 1.0

    indices = range(run_skip, run_skip + run_limit, 1)
    random.shuffle(indices)
    count = 0
    for i in indices:

        gtruth = np.argmax(labels[i, :])
        predicted = model.gen(data[i:i + 1, :])[0]
        print gtruth, predicted

        if predicted == gtruth:
            average_error = (1 - eval_coeff) * average_error
        else:
            average_error = eval_coeff + (1 - eval_coeff) * average_error
        # average_error = learning_coeff * (np.sum((label_ - labels[i, :])**2)) + (1 - learning_coeff) * average_error

        print "sample ", count, " error: ", average_error
        error_graph.append(average_error)
        count = count + 1

        model.init()

        for j in xrange(100):
            _, loss = model.learn(data[i:i + 1, :], labels[i:i + 1, :])
            if loss < 1e-3:
                break
        print loss

        model.update()
        model.collect(data[i:i + 1, :], learning_coeff)

        # model.debug_test()
        if count % 100 == 0:
            model.save()

    model.save()
    with open("./artifacts/" + ','.join(str(x) for x in layer_sizes) + ".txt", "w") as output:
        for v in error_graph:
            output.write("%s\n" % str(v))

    plt.plot(xrange(run_skip, run_skip + run_limit, 1), error_graph)
    plt.ylabel('average error')
    plt.show()
