import logging

import numpy as np
import tensorflow as tf
from keras.engine import Input
from keras.engine import Model
from keras.layers import Convolution2D, MaxPooling2D, Dense, Reshape, UpSampling2D, K, Lambda
from keras import metrics


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
    elif activation == "tanh":
        out = tf.nn.tanh(out, name='' + name + '_tanh')

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
        self.latent_shape = conf['latent_shape']
        self.keep_prob = tf.placeholder(tf.float32)  # For dropout
        self.dynamics_input = tf.placeholder("float32", [None, self.latent_shape],
                                             name="latent_input")
        self.dynamics_input_prev = tf.placeholder("float32", [None, self.latent_shape],
                                                  name="latent_input_prev")
        self.action_input = tf.placeholder("float32", [None, self.num_actions],
                                           name="action_input")
        self.autoencoder_input_ph = None
        self.autoencoder_input = None
        self.autoencoder_output = None
        self.encoder_output = None
        self.decoder_output = None
        self.decoder_input = None
        self.latent_prediction = None

        with tf.device(self.device):
            with tf.name_scope(self.name):
                self.one_over_emulators = 1.0 / self.emulator_counts
                self.input_ph = tf.placeholder(tf.uint8, [None, 84, 84, 4], name='input')
                self.selected_action_ph = tf.placeholder("float32", [None, self.num_actions], name="selected_action")
                self.input = tf.scalar_mul(1.0 / 255.0, tf.cast(self.input_ph, tf.float32))

                self.autoencoder_input_ph = tf.placeholder(tf.uint8, [None, 84, 84, 1], name='autoencoder_input')
                self.autoencoder_input = tf.scalar_mul(1.0 / 255.0, tf.cast(self.autoencoder_input_ph, tf.float32))
                self.decoder_input = tf.placeholder(tf.float32, [None, self.latent_shape], name='decoder_input')

                # This class should never be used, must be subclassed

                # The output layer
                self.output = None

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

        with tf.device(self.device):
            with tf.name_scope(self.name):
                # NIPS network
                _, _, conv1 = conv2d('conv1', self.input, 16, 8, 4, 4)
                _, _, conv2 = conv2d('conv2', conv1, 32, 4, 16, 2)
                _, _, fc3 = fc('fc3', flatten(conv2), 256, activation="relu")
                self.output = fc3

                # Encoder
                e1 = Convolution2D(48, 4, 4, activation='relu', border_mode='same', name='e1')(self.autoencoder_input)
                e2 = MaxPooling2D((2, 2), border_mode='same', name='e2')(e1)
                e3 = Convolution2D(64, 2, 2, activation='relu', border_mode='same', name='e3')(e2)
                e4 = MaxPooling2D((3, 3), border_mode='same', name='e4')(e3)
                e5 = Dense(self.latent_shape, activation='relu', name='e5')(flatten(e4))
                self.encoder_output = e5

                # Decoder layers
                d1 = Dense(3136, activation='relu', name='d1')
                d2 = Reshape((14, 14, 16), name='d2')
                d3 = Convolution2D(64, 2, 2, activation='relu', border_mode='same', name='d3')
                d4 = UpSampling2D((3, 3), name='d4')
                d5 = Convolution2D(48, 4, 4, activation='relu', border_mode='same', name='d5')
                d6 = UpSampling2D((2, 2), name='d6')
                d7 = Convolution2D(1, 4, 4, activation='relu', border_mode='same', name='d7')

                # Full autoencoder
                d1_full = d1(self.encoder_output)
                d2_full = d2(d1_full)
                d3_full = d3(d2_full)
                d4_full = d4(d3_full)
                d5_full = d5(d4_full)
                d6_full = d6(d5_full)
                d7_full = d7(d6_full)

                # Only decoding
                d1_decoder = d1(self.decoder_input)
                d2_decoder = d2(d1_decoder)
                d3_decoder = d3(d2_decoder)
                d4_decoder = d4(d3_decoder)
                d5_decoder = d5(d4_decoder)
                d6_decoder = d6(d5_decoder)
                d7_decoder = d7(d6_decoder)

                self.decoder_output = d7_decoder
                self.autoencoder_output = d7_full

                # Prediction on latent space
                prediction_input = tf.concat([self.dynamics_input_prev, self.dynamics_input, self.action_input], axis=1,
                                             name='prediction_input_concat')
                _, _, pred1 = fc('pred1', prediction_input, 500, activation="tanh")
                pred1 = tf.nn.dropout(pred1, keep_prob=self.keep_prob, name='pred1_drop')
                _, _, pred2 = fc('pred2', pred1, 500, activation="tanh")
                pred2 = tf.nn.dropout(pred2, keep_prob=self.keep_prob, name='pred2_drop')
                _, _, pred3 = fc('pred3', pred2, self.latent_shape, activation="tanh")
                self.latent_prediction = pred3

                # VAE test
                # batch_size = 32
                # intermediate_dim = 3136
                # original_dim = 7056
                # epsilon_std = 1.0
                #
                # x = Input(batch_shape=(batch_size, 84, 84, 1))
                # h = Dense(intermediate_dim, activation='relu')(flatten(x))
                # z_mean = Dense(self.latent_shape)(h)
                # z_log_var = Dense(self.latent_shape)(h)
                #
                # def sampling(args):
                #     z_mean, z_log_var = args
                #     epsilon = K.random_normal(shape=(batch_size, self.latent_shape), mean=0., stddev=epsilon_std)
                #     return z_mean + K.exp(z_log_var / 2) * epsilon
                #
                # z = Lambda(sampling)([z_mean, z_log_var])
                #
                # # we instantiate these layers separately so as to reuse them later
                # decoder_h = Dense(intermediate_dim, activation='relu')
                # decoder_mean = Dense(original_dim, activation='sigmoid')
                # h_decoded = decoder_h(z)
                # x_decoded_mean = decoder_mean(h_decoded)
                #
                # def vae_loss(x, x_decoded_mean):
                #     xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
                #     kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
                #     return xent_loss + kl_loss
                #
                # vae = Model(x, x_decoded_mean)
                # vae.compile(optimizer='rmsprop', loss=vae_loss)
                #
                # encoder = Model(x, z_mean)
                # decoder_input = Input(shape=(self.latent_shape,))
                # _h_decoded = decoder_h(decoder_input)
                # _x_decoded_mean = decoder_mean(_h_decoded)
                # generator = Model(decoder_input, _x_decoded_mean)


