import numpy as np
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.models import Model, Sequential
from matplotlib import pyplot as plt
import tensorflow as tf

"""
Tests different architectures for (conv) autoencoders alone and in combination with a transition prediction network.
"""


def build_deep_autoencoder():
    """
    Simple fc autoencoder
    """
    input_img = Input(shape=(7056,))
    encoded = Dense(2048, activation='relu')(input_img)
    encoded = Dense(1024, activation='relu')(encoded)
    encoded = Dense(256, activation='relu')(encoded)

    decoded = Dense(1024, activation='relu')(encoded)
    decoded = Dense(2048, activation='relu')(decoded)
    decoded = Dense(7056, activation='relu')(decoded)

    autoencoder = Model(input=input_img, output=decoded)
    autoencoder.compile(optimizer='adam', metrics=['mse'], loss='mse')
    autoencoder.summary()
    return autoencoder


def build_conv_combo_autoencoder():
    """
    Convolutional autoencoder with fc layers to the embedded space layer which is consists of 512 nodes
    """
    input_img = Input(shape=(84, 84, 4))

    x = Convolution2D(48, 4, 4, activation='relu', border_mode='same', name='c1')(input_img)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Convolution2D(48, 4, 4, activation='relu', border_mode='same', name='c2')(x)
    x = MaxPooling2D((3, 3), border_mode='same')(x)
    x = Flatten()(x)
    encoded = Dense(512, activation='relu')(x)

    encoded_input = Input((512,))
    d1 = Dense(9408, activation='relu')(encoded_input)
    d2 = Reshape((14, 14, 48))(d1)
    d3 = Convolution2D(48, 4, 4, activation='relu', border_mode='same', name='c5')(d2)
    d4 = UpSampling2D((3, 3))(d3)
    d5 = Convolution2D(48, 4, 4, activation='relu', border_mode='same', name='c6')(d4)
    d6 = UpSampling2D((2, 2))(d5)
    decoded = Convolution2D(4, 4, 4, activation='relu', border_mode='same', name='c9')(d6)

    encoder = Model(input=input_img, output=encoded, name='conv_encoder')
    decoder = Model(input=encoded_input, output=decoded, name='conv_decoder')

    autoencoder = Sequential(name='full_conv_autoencoder')
    autoencoder.add(encoder)
    autoencoder.add(decoder)

    encoder.compile(optimizer='adam', loss='mse')
    encoder.summary()
    decoder.compile(optimizer='adam', loss='mse')
    decoder.summary()
    autoencoder.compile(optimizer='adam', metrics=['mse'], loss='mse')
    autoencoder.summary()
    return autoencoder, encoder, decoder


def build_conv_autoencoder():
    """
    Regular convolutional autoencoder
    """
    input_img = Input(shape=(84, 84, 1))

    x = Convolution2D(32, 7, 7, activation='relu', border_mode='same', name='c1')(input_img)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Convolution2D(48, 4, 4, activation='relu', border_mode='same', name='c2')(x)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Convolution2D(32, 4, 4, activation='relu', border_mode='same', name='c3')(x)
    x = MaxPooling2D((1, 1), border_mode='same')(x)
    x = Convolution2D(48, 3, 3, activation='relu', border_mode='same', name='c4')(x)
    x = MaxPooling2D((1, 1), border_mode='same')(x)
    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same', name='c10')(x)
    encoded = MaxPooling2D((3, 3), border_mode='same')(x)

    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same', name='c5')(encoded)
    x = UpSampling2D((3, 3))(x)
    x = Convolution2D(32, 4, 4, activation='relu', border_mode='same', name='c6')(x)
    x = UpSampling2D((1, 1))(x)
    x = Convolution2D(48, 4, 4, activation='relu', border_mode='same', name='c7')(x)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(32, 7, 7, activation='relu', border_mode='same', name='c8')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Convolution2D(1, 4, 4, activation='sigmoid', border_mode='same', name='c9')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', metrics=['mse'], loss='mse')
    autoencoder.summary()
    return autoencoder


def build_full_conv_autoencoder_new(z_shape):
    encoder = Sequential(name="encoder")
    encoder.add(Convolution2D(32, 8, 8, border_mode='same', activation='relu',
                              input_shape=(84, 84, 4)))  # TODO change to parameters
    encoder.add(MaxPooling2D((2, 2), border_mode='same'))
    encoder.add(Convolution2D(64, 4, 4, border_mode='same', activation='relu'))
    encoder.add(MaxPooling2D((3, 3), border_mode='same'))
    encoder.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    encoder.add(Flatten())
    encoder.add(Dense(z_shape, activation='relu'))
    encoder.summary()

    decoder = Sequential(name="decoder")
    decoder.add(Dense(784, input_shape=(z_shape,), activation='relu'))
    decoder.add(Reshape((14, 14, 4)))
    decoder.add(Convolution2D(48, 3, 3, border_mode='same', activation='relu'))
    decoder.add(UpSampling2D((3, 3)))
    decoder.add(Convolution2D(48, 4, 4, border_mode='same', activation='relu'))
    decoder.add(UpSampling2D((2, 2)))
    decoder.add(Convolution2D(4, 4, 4, border_mode='same', activation='relu'))
    decoder.summary()

    autoencoder = Sequential(name='autoencoder')
    autoencoder.add(encoder)
    autoencoder.add(decoder)
    autoencoder.compile(optimizer='adam', metrics=['mse'], loss='mse')
    autoencoder.summary()
    return autoencoder, encoder, decoder


def build_full_conv_autoencoder():
    """
    A convolutional autoencoder that takes colored images (not grayscale)
    """
    input_img = Input(shape=(84, 84, 3))

    x = Convolution2D(48, 8, 8, activation='relu', border_mode='same', name='c1')(input_img)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Convolution2D(32, 4, 4, activation='relu', border_mode='same', name='c2')(x)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same', name='c3')(x)
    encoded = MaxPooling2D((3, 3), border_mode='same')(x)

    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same', name='c4')(encoded)
    x = UpSampling2D((3, 3))(x)
    x = Convolution2D(32, 4, 4, activation='relu', border_mode='same', name='c5')(x)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(48, 8, 8, activation='relu', border_mode='same', name='c6')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Convolution2D(3, 4, 4, activation='sigmoid', border_mode='same', name='c7')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', metrics=['mse'], loss='mse')
    autoencoder.summary()
    return autoencoder


def build_deep_predictor(state_size, action_size):
    """
    A fc transition prediction network that takes a compressed latent space and a one-hot vector of an action to predict
    the latent space for the next state
    :param state_size: latent space size
    :param action_size: number of valid actions
    """
    n_input = state_size+action_size
    n_hidden_1 = 1024
    n_hidden_2 = 1024

    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, state_size])
    keep_prob = tf.placeholder(tf.float32)  # For dropout

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, state_size]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([state_size]))
    }

    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.dropout(layer_1, keep_prob)  # Dropout layer
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    layer_2 = tf.nn.dropout(layer_2, keep_prob)  # Dropout layer

    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']

    loss_function = tf.reduce_mean(tf.pow(y - out_layer, 2))
    optimizer = tf.train.AdamOptimizer().minimize(loss_function)

    return out_layer, loss_function, optimizer, keep_prob, x, y

    # concated_input = Input(shape=(state_size + action_size,))
    # output = Dense(1024, activation='relu', name='p_d1')(concated_input)
    # output = Dense(1024, activation='relu', name='p_d2')(output)
    # output = Dense(state_size, activation='relu', name='p_dout')(output)
    #
    # predictor = Model(input=concated_input, output=output, name='transition_model')
    # predictor.compile(optimizer='adam', metrics=['mse'], loss='mse')
    # predictor.summary()
    # return predictor


def encode_to_samples(encoder, X_train, actions, z_shape, num_actions, print_labels=True, split_inputs=False):
    """
    Takes an encoder in and compresses X_train into latent representations and concatenates the representations with
    the selected action from actions.
    :param encoder: the encoder to output a latent representation from states
    :param X_train: the observations (states)
    :param actions: the selected actions for each observation (one-hot vector for each action)
    :param z_shape: shape of latent space
    :param split_inputs: if true --> concatenate actions with latent representation
    :return: training samples for the transition prediction model
    """
    arr = encoder.predict(X_train)
    arr = arr.reshape(len(X_train), z_shape)
    a_one_hot = np.eye(num_actions)[actions]
    if print_labels:
        if not split_inputs:
            x_samples = np.concatenate((arr, a_one_hot), axis=1)
            x_samples = x_samples[:-1]
        else:
            x_samples = arr[:-1]
        y_samples = arr[1:len(arr)]
        return np.array(x_samples), np.array(y_samples)
    else:
        if not split_inputs:
            return np.concatenate((arr, a_one_hot), axis=1)
        else:
            return arr


def train_predictor(predictor, X_train, Y_train, nb_epoch=5, batch_size=32):
    """
    Train a transition prediction network
    """
    history = predictor.fit(X_train, Y_train,
                            nb_epoch=nb_epoch,
                            batch_size=batch_size,
                            shuffle=True,
                            verbose=2)
    del X_train

    return np.mean(history.history['loss'])


def train_autoencoder(autoencoder, X_train, X_test, nb_epoch=5, batch_size=32, nb_batches=None, nb_examples=0, verbose=2):
    """
    Trains an autoencoder
    """
    if nb_batches is not None:
        X_train = X_train[:nb_batches * batch_size]
    history = autoencoder.fit(X_train, X_train,
                              nb_epoch=nb_epoch,
                              batch_size=batch_size,
                              shuffle=True,
                              # validation_data=(X_test, X_test),
                              # callbacks=[TensorBoard(log_dir='/tmp/autoencoder', histogram_freq=1)],
                              verbose=verbose)
    del X_train

    plot_examples_autoencoder(autoencoder, X_test, nb_examples=nb_examples)

    return np.max(history.history['loss'])


def predict_autoencoder(autoencoder, X_test, show_examples=True):
    """
    Special method for comparing two observations (used for comparing room1 observation to never seen room2 observation
     in Montezuma's Revenge)
    """
    res_1 = []
    res_2 = []
    result = autoencoder.evaluate(np.array([X_test[0]]), np.array([X_test[0]]), batch_size=1, verbose=0)
    res_1.append(result[1])
    result = autoencoder.evaluate(np.array([X_test[1]]), np.array([X_test[1]]), batch_size=1, verbose=0)
    res_2.append(result[1])

    if show_examples:
        plot_examples_autoencoder(autoencoder, X_test, nb_examples=len(X_test))

    return res_1, res_2


def plot_examples_autoencoder(autoencoder, X_test, nb_examples=10):
    """
    Plots nb_examples examples with original images and restored images from an autoencoder
    """

    if nb_examples == 0:
        return

    decoded_imgs = autoencoder.predict(X_test)

    n = nb_examples
    plt.figure(figsize=(nb_examples * 2, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(X_test[i].reshape(84, 84))
        # plt.imshow(X_test[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(84, 84))
        # plt.imshow(decoded_imgs[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def plot_images(imgs_org, imgs_pred, nb_examples=10):
    """
    Plots nb_examples images with original image and a predicted (restored) image
    """
    if nb_examples == 0:
        return

    n = nb_examples
    plt.figure(figsize=(nb_examples * 2, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(imgs_org[i].reshape(84, 84))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(imgs_pred[i].reshape(84, 84))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
