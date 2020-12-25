#!/usr/bin/env python
import numpy as np
import tensorflow as tf
import imageio
import PIL
from train import inference



if __name__ == "__main__":

    weightdir = "./weight/"
    pb_path = "./weight/model2.pb"

    x = tf.placeholder(tf.float64, [None, 3], name='input')
    x = tf.cast(x, tf.float32)
    y = inference(x, 3, 10, 1)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=5)

    ckpt = tf.train.get_checkpoint_state(weightdir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored from ", ckpt.model_checkpoint_path)

    # save pb
    constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])
    with tf.gfile.FastGFile(pb_path, mode='wb') as f:
        f.write(constant_graph.SerializeToString())
    print("%d ops in the final graph." % len(constant_graph.node))









