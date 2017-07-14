import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import argparse
import numpy as np
import os
import src.matter as matter

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

    print "Dataset:", data.shape
    print "Labels:", labels.shape

    # world_size = (60, 60)
    # past_steps = 2
    # components = 2
    # component_size = 30
    # belief_depth = 20
    # total_maps = 5 if not args.total else args.total

    # print "-----------------------"
    # print "world size: ", world_size
    # print "past steps: ", past_steps
    # print "components: ", components
    # print "component size: ", component_size
    # print "memory depth: ", belief_depth
    # print "total maps: ", total_maps
    # print "-----------------------"

    # sess = tf.Session()
    # machine = core.Machine(sess, world_size[0] * world_size[1], past_steps, components, component_size, belief_depth)
    # sess.run(tf.global_variables_initializer())

    # if not args.reset:
    #     machine.load_session("./artifacts/demo")

    # if not args.gen:

    #     for m in xrange(total_maps):
    #         frames = world.get_valid_data(world_size, world_size[0] / 10, map_complexity=6, length_modifier=0.2)
    #         machine.reset_memory()
    #         for i in xrange(0, frames.shape[0]):
    #             pasts = util.prepare_data(frames, i - past_steps, i)
    #             input_data = util.prepare_data(frames, i, i + 1)
    #             print "-----------"
    #             # learn and save model
    #             machine.learn(input_data, pasts, 20, "./artifacts/demo")
    # else:

    #     generated_frames = []
    #     frames = world.get_valid_data(world_size, world_size[0] / 10, map_complexity=6, length_modifier=0.2)
    #     machine.reset_memory()
    #     pasts = util.prepare_data(frames, 0 - past_steps, 0)
    #     input_data = util.prepare_data(frames, 0, 0 + 1)

    #     # Learn the target frame
    #     machine.learn(input_data, pasts, 100)

    #     for i in xrange(1, frames.shape[0]):
    #         pasts = util.prepare_data(frames, i - past_steps, i)
    #         input_data = util.prepare_data(frames, i, i + 1)
    #         # generate thoughts
    #         gen = machine.generate_thought(pasts)
    #         # and also memorize the generated thoughts, but not save
    #         machine.learn(gen, pasts, 100)

    #         generated_frames.append(gen)

    #     artifact_path = os.path.dirname(os.path.abspath(__file__)) + "/artifacts/"
    #     world.toGif(world.to_numpy(generated_frames, world_size), artifact_path + "sample_path.gif")
