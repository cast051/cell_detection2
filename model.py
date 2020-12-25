import tensorflow as tf
import tensorflow.contrib.slim as slim
from base.base_block import BaseBlock


class Model(BaseBlock):
    def __init__(self, config):
        self.is_training = config.is_training
        # define variable
        self.image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input")
        if self.is_training:
            self.annotation = tf.placeholder(tf.float32, shape=[None, None, None, 1], name="annotation")

        self.num_classes = config.num_classes
        self.layers_down, self.layers_up, self.batch_norm_params = self.net_config()
        self.logits = None
        self.y = None

    def net_config(self):
        layers_down = [
            [8, 8, 3, 2, tf.nn.relu6, True, 8],
            [8, 16, 3, 2, tf.nn.relu6, False, 32],
            [16, 16, 3, 1, tf.nn.relu6, False, 48],
            [16, 32, 3, 2, self.hard_swish, True, 64],
            [32, 32, 3, 1, self.hard_swish, True, 128],
        ]
        layers_up = [
            [16, 16, 3, 1, tf.nn.relu6, True, 32],
            [8, 8, 3, 1, tf.nn.relu6, True, 16],
            [8, 8, 3, 1, tf.nn.relu6, True, 8],
        ]
        batch_norm_params = {
            'decay': 0.99,
            'epsilon': 0.01,
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
            'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
            'is_training': self.is_training,
            'scale':True,
        }
        return layers_down, layers_up, batch_norm_params

    def inference(self, scope='inference'):
        up_node = []
        with tf.variable_scope(scope):
            with slim.arg_scope([slim.conv2d,slim.conv2d_transpose],
                                weights_regularizer=slim.l2_regularizer(0.00001),
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.001),
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=self.batch_norm_params):
                # conv
                out = slim.conv2d(self.image, 8, [3, 3], activation_fn=tf.nn.relu, stride=1, scope="conv1")
                # down neck
                for idx, (in_size, out_size, kernel_size, stride, activation_fn, se, expand_size) in enumerate(
                        self.layers_down):
                    if stride == 2:
                        up_node.append(out)
                    out = self.mobilenet_v3_block(out, in_size, expand_size, out_size, [kernel_size, kernel_size],
                                                  self.batch_norm_params,
                                                  stride=stride, activation_fn=activation_fn, ratio=4, se=se,
                                                  scope="bneck{}".format(idx))

                # up neck
                for idx, (in_size, out_size, kernel_size, stride, activation_fn, se, expand_size) in enumerate(
                        self.layers_up):
                    out = slim.conv2d_transpose(out, out_size, kernel_size, stride=2, scope="transpose{}".format(idx))
                    out = tf.add(out, up_node[len(up_node) - idx - 1], name="fuse{}".format(idx))
                    out = self.mobilenet_v3_block(out, in_size, expand_size, out_size, [kernel_size, kernel_size],
                                                  self.batch_norm_params, stride=stride, activation_fn=activation_fn,
                                                  ratio=4, se=se, scope="upbneck{}".format(idx))
                # conv
                out = slim.conv2d(out, self.num_classes, [1, 1], activation_fn=None, scope="conv1x1")
                self.logits = out
                out = tf.nn.sigmoid(out)
        self.y = tf.identity(out, name='net_output')
        # return out,logits


# test
from config import get_config

if __name__ == '__main__':
    config=get_config(is_training=True)
    model=Model(config=config)
    model.inference()
    print(model.y.shape)
    print(model.logits.shape)



