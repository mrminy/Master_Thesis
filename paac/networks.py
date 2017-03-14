import tensorflow as tf
import logging
import numpy as np


def flatten(_input):
    shape = _input.get_shape().as_list()
    dim = shape[1] * shape[2] * shape[3]
    return tf.reshape(_input, [-1, dim], name='_flattened')


def conv2d(name, _input, filters, size, channels, stride, padding='VALID', init="torch"):
    w = conv_weight_variable([size, size, channels, filters], name + '_weights', init=init)
    b = conv_bias_variable([filters], size, size, channels, name + '_biases', init=init)
    conv = tf.nn.conv2d(_input, w, strides=[1, stride, stride, 1], padding=padding, name=name + '_convs')
    out = tf.nn.relu(tf.add(conv, b),
                     name='' + name + '_activations')
    return w, b, out


def maxpool2d(name, _input, pool_size):
    return tf.nn.max_pool(_input, ksize=[1, pool_size, pool_size, 1], strides=[1, pool_size, pool_size, 1],
                          padding='SAME', name=name + '_maxpool')


def conv_weight_variable(shape, name, init="torch"):
    if init == "glorot_uniform":
        receptive_field_size = np.prod(shape[:2])
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
        d = np.sqrt(6. / (fan_in + fan_out))
    else:
        w = shape[0]
        h = shape[1]
        input_channels = shape[2]
        d = 1.0 / np.sqrt(input_channels * w * h)

    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name, dtype='float32')


def conv_bias_variable(shape, w, h, input_channels, name, init="torch"):
    if init == "glorot_uniform":
        initial = tf.zeros(shape)
    else:
        d = 1.0 / np.sqrt(input_channels * w * h)
        initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name, dtype='float32')


def fc(name, _input, output_dim, activation="relu", init="torch"):
    input_dim = _input.get_shape().as_list()[1]
    w = fc_weight_variable([input_dim, output_dim],
                           name + '_weights', init=init)
    b = fc_bias_variable([output_dim], input_dim,
                         '' + name + '_biases', init=init)
    out = tf.add(tf.matmul(_input, w), b, name=name + '_out')

    if activation == "relu":
        out = tf.nn.relu(out, name='' + name + '_relu')

    return w, b, out


def fc_weight_variable(shape, name, init="torch"):
    if init == "glorot_uniform":
        fan_in = shape[0]
        fan_out = shape[1]
        d = np.sqrt(6. / (fan_in + fan_out))
    else:
        input_channels = shape[0]
        d = 1.0 / np.sqrt(input_channels)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name, dtype='float32')


def fc_bias_variable(shape, input_channels, name, init="torch"):
    if init == "glorot_uniform":
        initial = tf.zeros(shape, dtype='float32')
    else:
        d = 1.0 / np.sqrt(input_channels)
        initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name, dtype='float32')


def softmax(name, _input, output_dim):
    input_dim = _input.get_shape().as_list()[1]
    w = fc_weight_variable([input_dim, output_dim], name + '_weights')
    b = fc_bias_variable([output_dim], input_dim, name + '_biases')
    out = tf.nn.softmax(tf.add(tf.matmul(_input, w), b), name=name + '_policy')
    return w, b, out


def log_softmax(name, _input, output_dim):
    input_dim = _input.get_shape().as_list()[1]
    w = fc_weight_variable([input_dim, output_dim], name + '_weights')
    b = fc_bias_variable([output_dim], input_dim, name + '_biases')
    out = tf.nn.log_softmax(tf.add(tf.matmul(_input, w), b), name=name + '_policy')
    return w, b, out


class Network(object):
    def __init__(self, conf):
        """ Initialize hyper-parameters, set up optimizer and network
        layers common across Q and Policy/V nets. """

        self.name = conf['name']
        self.num_actions = conf['num_actions']
        self.clip_norm = conf['clip_norm']
        self.clip_norm_type = conf['clip_norm_type']
        self.emulator_counts = conf['emulator_counts']
        self.device = conf['device']

        # Vars used in dynamics model
        self.latent_size = 512
        self.keep_prob = tf.placeholder(tf.float32)  # For dropout
        self.dynamics_input = None

        with tf.device(self.device):
            with tf.name_scope(self.name):
                self.one_over_emulators = 1.0 / self.emulator_counts
                self.input_ph = tf.placeholder(tf.uint8, [None, 84, 84, 4], name='input')
                self.selected_action_ph = tf.placeholder("float32", [None, self.num_actions], name="selected_action")
                self.input = tf.scalar_mul(1.0 / 255.0, tf.cast(self.input_ph, tf.float32))

                # This class should never be used, must be subclassed

                # The output layer
                self.output = None
                self.latent_prediction = None

    def init(self, debugging_folder, session):
        import os
        if not os.path.exists(debugging_folder + '/checkpoints/'):
            os.makedirs(debugging_folder + '/checkpoints/')

        last_saving_step = 0
        self.saver = tf.train.Saver()

        with tf.device('/cpu:0'):
            # Initialize network parameters
            path = tf.train.latest_checkpoint(debugging_folder + '/checkpoints/')
            if path is None:
                logging.info('Initializing all variables')
                session.run(tf.global_variables_initializer())
            else:
                logging.info('Restoring variables from previous run')
                self.saver.restore(session, path)
                last_saving_step = int(path[path.rindex('-') + 1:])
        return last_saving_step


class NIPSNetwork(Network):
    def __init__(self, conf):
        super(NIPSNetwork, self).__init__(conf)

        with tf.device(self.device):
            with tf.name_scope(self.name):
                _, _, conv1 = conv2d('conv1', self.input, 16, 8, 4, 4)

                _, _, conv2 = conv2d('conv2', conv1, 32, 4, 16, 2)

                _, _, fc3 = fc('fc3', flatten(conv2), 256, activation="relu")

                self.output = fc3


class NatureNetwork(Network):
    def __init__(self, conf):
        super(NatureNetwork, self).__init__(conf)

        with tf.device(self.device):
            with tf.name_scope(self.name):
                _, _, conv1 = conv2d('conv1', self.input, 32, 8, 4, 4)

                _, _, conv2 = conv2d('conv2', conv1, 64, 4, 32, 2)

                _, _, conv3 = conv2d('conv3', conv2, 64, 3, 64, 1)

                _, _, fc4 = fc('fc4', flatten(conv3), 512, activation="relu")

                self.output = fc4


class DynamicsNetwork(Network):
    def __init__(self, conf):
        super(DynamicsNetwork, self).__init__(conf)

        self.latent_size = 512

        with tf.device(self.device):
            with tf.name_scope(self.name):
                # Encoder
                _, _, conv1 = conv2d('conv1', self.input, 32, 8, 4, 4)
                _, _, conv2 = conv2d('conv2', conv1, 64, 4, 32, 2)
                _, _, conv3 = conv2d('conv3', conv2, 64, 3, 64, 1)
                _, _, fc4 = fc('fc4', flatten(conv3), self.latent_size, activation="relu")
                self.output = fc4

                # Prediction on latent space
                self.dynamics_input = tf.placeholder("float32", [None, self.latent_size + self.num_actions],
                                                     name="transition_prediction_input")
                _, _, pred1 = fc('pred1', self.dynamics_input, 1024)
                pred1 = tf.nn.dropout(pred1, keep_prob=self.keep_prob, name='pred1_drop')
                _, _, pred2 = fc('pred2', pred1, 1024)
                pred2 = tf.nn.dropout(pred2, keep_prob=self.keep_prob, name='pred1_drop')
                _, _, pred3 = fc('pred3', pred2, self.latent_size)
                self.latent_prediction = pred3

                # Decoder
                # _, _, deconv1 = conv2d('deconv1', fc4.reshape(), 64, 3, 64, 1)
                # _, _, deconv2 = conv2d('deconv2', conv1, 64, 4, 32, 2)
                # _, _, deconv3 = conv2d('deconv3', conv2, 32, 8, 4, 4)
                # _, _, deconv4 = conv2d('deconv4', conv2, 4, 84, 1, 1)
                # self.autoencoder_output = deconv4
