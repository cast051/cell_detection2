#!/usr/bin/env python
import numpy as np
import tensorflow as tf
import imageio
import os

os.environ['CUDA_VISIBLE_DEVICES']='7'

def inference(x, in_num, hidden_num, out_num):
    w1 = tf.Variable(tf.truncated_normal([in_num, hidden_num], stddev=0.1))
    b1 = tf.Variable(tf.zeros([hidden_num]))
    w2 = tf.Variable(tf.zeros([hidden_num, out_num]))
    b2 = tf.Variable(tf.zeros([out_num]))
    layer1 = tf.nn.relu(tf.add(tf.matmul(x, w1), b1))
    out = tf.nn.sigmoid(tf.add(tf.matmul(layer1, w2), b2))
    out = tf.identity(out, name='output')
    return out


if __name__ == "__main__":

    weightdir = "./weight/"
    model = "test"  # train   |   test  |  test2
    pb_path = "./weight/model2.pb"
    img_path="data.png"
    learningrate = 0.003

    color = imageio.imread(img_path)
    x_data = np.zeros((41, 3), dtype=np.int)
    for i in range(41):
        x_data[i, :] = color[0, 12 * i, :3]
    y_data = np.linspace(0, 1, 41)
    y_data = np.expand_dims(y_data, 1)

    y_ = tf.placeholder(tf.float32, [None, 1])
    x = tf.placeholder(tf.float64, [None, 3], name='input')
    x = tf.cast(x, tf.float32)

    y = inference(x, 3, 10, 1)

    loss = tf.losses.mean_squared_error(y, y_)
    optimizer = tf.train.AdamOptimizer(learningrate).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=5)

    ckpt = tf.train.get_checkpoint_state(weightdir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored from ", ckpt.model_checkpoint_path)

    if model == "train":
        for i in range(1000000):
            feed = {x: x_data, y_: y_data}
            loss_val, _, out = sess.run([loss, optimizer, y], feed_dict=feed)
            if i % 200 == 0:
                # print(i," ","loss: ",loss_val," | ",out,"  ",y_data," | ")
                print("loss: ", loss_val)
            if i % 2000 == 0:
                print("Saved model ")
                # save model
                saver.save(sess, weightdir + "model_.ckpt", i)


    elif model == "test":
        feed = {x: x_data}
        out = sess.run([y], feed_dict=feed)
        for i in range(out[0].shape[0]):
            print(out[0][i], "  |  ", y_data[i])

    elif model == "test2":

        test_data = np.zeros((1, 3), dtype=np.int)
        test_data[0] = [255, 106, 106]

        feed = {x: test_data}
        out = sess.run([y], feed_dict=feed)
        for i in range(out[0].shape[0]):
            print(out[0][i], "  |  ", y_data[i])






