import numpy as np
import random
from math import sin
import tensorflow as tf
import matplotlib.pyplot as plt
import time


def reg_function(x, dist):
    return x + sin(4 * (x + dist)) + sin(13 * (x + dist)) + dist


def generate_samples(n):
    dist = np.random.normal(0.0, 0.03, n)
    xs = []
    ys = []

    for i in range(0, n):
        # if random.random() > 0.5:
        #     x = random.uniform(0.8, 2.)
        # else:
        #     x = random.uniform(-2.5, -1.)
        x = random.uniform(-4., 3.)
        xs.append(x)
    xs.sort()

    for i in range(0, n):
        y = reg_function(xs[i], dist[i])
        ys.append(y)

    # Add noisy points
    xs.append(5.)
    xs.append(5.)
    y = reg_function(xs[-1], 0.0)
    ys.append(y + 6)
    ys.append(y - 6)
    return np.array(xs), np.array(ys)


def generate_x_samples(n_samples=10000, min_val=-10, max_val=10):
    X_pred = []
    for i in range(0, n_samples):
        X_pred.append([random.uniform(min_val, max_val)])
    X_pred.sort()
    return np.array(X_pred)


def plot_uncertainty_regression(X, Y, X_pred, Y_pred_mean, Y_pred_std):
    Y_pred_mean_min = Y_pred_mean - Y_pred_std
    Y_pred_mean_max = Y_pred_mean + Y_pred_std
    Y_pred_mean_min_double = Y_pred_mean - 2 * Y_pred_std
    Y_pred_mean_max_double = Y_pred_mean + 2 * Y_pred_std

    # Generate the correct line
    Y_target = []
    for x in X_pred:
        Y_target.append(reg_function(x, 0.0))
    Y_target = np.array(Y_target)

    plt.plot(X, Y, 'ro', X_pred, Y_pred_mean, X_pred, Y_target)
    plt.fill_between(X_pred.flatten(), Y_pred_mean_max.flatten(), Y_pred_mean_min.flatten(), edgecolor='blue',
                     facecolor='blue', alpha=0.5)
    plt.fill_between(X_pred.flatten(), Y_pred_mean_max_double.flatten(), Y_pred_mean_min_double.flatten(),
                     edgecolor='green', facecolor='green', alpha=0.3)
    plt.axis([-10, 10, -15, 15])
    plt.show()


def multilayer_perceptron_heads(x, weights, biases, keep, head):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.dropout(layer_1, keep_prob=keep)

    layer_head = tf.add(tf.matmul(layer_1, weights[head]), biases['b' + head])
    layer_head = tf.nn.relu(layer_head)
    layer_head = tf.nn.dropout(layer_head, keep_prob=keep)

    # Output layer with linear activation
    out_layer = tf.add(tf.matmul(layer_head, weights['out']), biases['out'])
    return out_layer


class MCDropout:
    def __init__(self, n_input, n_output, learning_rate, X_train, Y_train):
        # tf Graph input
        self.x = tf.placeholder("float", [None, n_input])
        self.y = tf.placeholder("float", [None, n_output])
        self.keep_prob = tf.placeholder("float")

        self.X_train = X_train
        self.Y_train = Y_train

        n_hidden_1 = 60
        n_hidden_2 = 60
        n_hidden_3 = 60
        n_hidden_4 = 60

        # Store layers weight & bias
        self.weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            # 'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
            # 'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
            'out': tf.Variable(tf.random_normal([n_hidden_2, n_output]))
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([n_hidden_2])),
            # 'b3': tf.Variable(tf.random_normal([n_hidden_3])),
            # 'b4': tf.Variable(tf.random_normal([n_hidden_4])),
            'out': tf.Variable(tf.random_normal([n_output]))
        }

        self.preds = self.build_network()

        # Define loss and optimizer
        delta = 0.000001
        self.cost = tf.reduce_mean(tf.square(tf.subtract(self.y, self.preds)))
        # + delta * tf.nn.l2_loss(
        #     weights['h1']) + delta * tf.nn.l2_loss(weights['h2']) + delta * tf.nn.l2_loss(
        #     weights['h3']) + delta * tf.nn.l2_loss(weights['h4']) + delta * tf.nn.l2_loss(weights['out']))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        # Initializing the variables
        self.init = tf.global_variables_initializer()

        # Launch the graph
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(self.init)

    def build_network(self):
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(self.x, self.weights['h1']), self.biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        layer_1 = tf.nn.dropout(layer_1, keep_prob=self.keep_prob)

        # Hidden layer with RELU activation
        layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
        layer_2 = tf.nn.dropout(layer_2, keep_prob=self.keep_prob)

        # layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
        # layer_3 = tf.nn.relu(layer_3)
        # layer_3 = tf.nn.dropout(layer_3, keep_prob=keep)
        #
        # layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
        # layer_4 = tf.nn.relu(layer_4)
        # layer_4 = tf.nn.dropout(layer_4, keep_prob=keep)

        # Output layer with linear activation
        out_layer = tf.add(tf.matmul(layer_2, self.weights['out']), self.biases['out'])
        return out_layer

    def train(self, training_epochs, batch_size=32, display_step=1, keep_prob=0.95):
        start_time = time.time()

        cost_history = []

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(len(self.X_train) / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                selected_x = [[self.X_train[i]]]
                selected_y = [[self.Y_train[i]]]
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = self.sess.run([self.optimizer, self.cost],
                                     feed_dict={self.x: selected_x, self.y: selected_y, self.keep_prob: keep_prob})
                # Compute average loss
                avg_cost += c / total_batch
            cost_history.append(avg_cost)
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
        print("Optimization finished after", time.time() - start_time)
        return cost_history

    def test(self, plot=True, n_stochastic_forward_passes=100, keep_prob=0.95):
        start_time = time.time()

        X_pred = generate_x_samples()

        # The next line can be used to see how it performs without any stochastic forward passes (TODO)
        # Y_pred_no_dropout = self.sess.run(self.preds, feed_dict={self.x: X_pred, self.keep_prob: 1.})

        Y_pred_heads = []
        if n_stochastic_forward_passes == 1:
            Y_pred_heads = self.sess.run(self.preds, feed_dict={self.x: X_pred, self.keep_prob: 1.})
            Y_pred_mean = np.mean(Y_pred_heads, axis=1)
            Y_pred_std = np.std(Y_pred_heads, axis=1)
        else:
            enumis = []  # Create copies of x_test
            for x_test in X_pred:
                for i in range(0, n_stochastic_forward_passes):
                    enumis.append(x_test)
            enumis = np.array(enumis)
            Y_pred_heads = self.sess.run(self.preds, feed_dict={self.x: enumis, self.keep_prob: keep_prob}).flatten()
            Y_pred_heads = Y_pred_heads.reshape((len(X_pred), n_stochastic_forward_passes))
            Y_pred_mean = np.mean(Y_pred_heads, axis=1).flatten()
            Y_pred_std = np.std(Y_pred_heads, axis=1).flatten()

        print("Time used on inference:", time.time() - start_time)
        if plot:
            plot_uncertainty_regression(self.X_train, self.Y_train, X_pred, Y_pred_mean, Y_pred_std)

    def close_session(self):
        self.sess.close()


class Bootstrap:
    def __init__(self, n_input, n_output, learning_rate, X_train, Y_train, nb_heads=4):
        # tf Graph input
        self.x = tf.placeholder("float", [None, n_input])
        self.y = tf.placeholder("float", [None, n_output])
        self.keep_prob = tf.placeholder("float")

        self.X_train = X_train
        self.Y_train = Y_train

        self.nb_heads = nb_heads
        self.H = np.random.randint(nb_heads, size=len(X_train))

        n_hidden_1 = 60
        n_hidden_2 = 60
        n_hidden_3 = 60
        n_hidden_4 = 60
        n_head = 1

        # Store layers weight & bias
        self.weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            # 'out': tf.Variable(tf.random_normal([n_hidden_2, n_output]))
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([n_hidden_2])),
            # 'out': tf.Variable(tf.random_normal([n_output]))
        }

        self.head_weights = {}
        self.head_biases = {}
        for i in range(0, nb_heads):
            str_index = 'head' + str(i)
            self.head_weights[str_index] = tf.Variable(tf.random_normal([n_hidden_2, n_head]))
            self.head_biases[str_index] = tf.Variable(tf.random_normal([n_head]))

        self.preds = []
        self.costs = []
        self.optimizers = []
        for i in range(0, nb_heads):
            self.preds.append(self.build_network(i))
            self.costs.append(tf.reduce_mean(tf.square(tf.subtract(self.y, self.preds[i]))))
            self.optimizers.append(tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.costs[i]))

        # Initializing the variables
        self.init = tf.global_variables_initializer()

        # Launch the graph
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(self.init)

    def build_network(self, head_index):
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(self.x, self.weights['h1']), self.biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        layer_1 = tf.nn.dropout(layer_1, keep_prob=self.keep_prob)

        # Hidden layer with RELU activation
        layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
        layer_2 = tf.nn.dropout(layer_2, keep_prob=self.keep_prob)

        # Output layer with linear activation
        out_layer = tf.add(tf.matmul(layer_2, self.head_weights['head' + str(head_index)]),
                           self.head_biases['head' + str(head_index)])
        return out_layer

    def train(self, training_epochs, batch_size=32, display_step=1, select_head_from_mask=False, keep_prob=0.95):
        start_time = time.time()

        cost_history = []

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(len(self.X_train) / batch_size)
            selected_head = random.randint(0, self.nb_heads - 1)  # Select random head

            # Loop over all batches
            for i in range(total_batch):
                selected_x = [[self.X_train[i]]]
                selected_y = [[self.Y_train[i]]]
                if select_head_from_mask:
                    selected_head = self.H[i]  # If using mask (each data point has a specific head)

                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = self.sess.run([self.optimizers[selected_head], self.costs[selected_head]],
                                     feed_dict={self.x: selected_x, self.y: selected_y, self.keep_prob: keep_prob})
                # Compute average loss
                avg_cost += c / total_batch
            cost_history.append(avg_cost)
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
        print("Optimization finished after", time.time() - start_time)
        return cost_history

    def test(self, plot=True, use_mc_dropout=False, n_stochastic_forward_passes=100, keep_prob=0.95):
        start_time = time.time()
        X_pred = generate_x_samples()

        Y_pred_heads = []
        if use_mc_dropout:
            # Combines mc dropout with bootstrap heads
            enumis = []
            for x_test in X_pred:
                for i in range(0, n_stochastic_forward_passes):
                    enumis.append(x_test)
            for head in range(self.nb_heads):
                Y_pred_heads.append(self.sess.run(self.preds[head], feed_dict={self.x: enumis,
                                                                               self.keep_prob: keep_prob}).flatten().reshape(
                    (len(X_pred), n_stochastic_forward_passes)))
            Y_pred_mean = np.mean(np.mean(Y_pred_heads, axis=2), axis=0).flatten()
            Y_pred_std = np.mean(np.std(Y_pred_heads, axis=2), axis=0).flatten()
        else:
            # Test bootstrap alone
            for head in range(self.nb_heads):
                Y_pred_heads.append(
                    self.sess.run(self.preds[head], feed_dict={self.x: X_pred, self.keep_prob: 1.}).flatten())
            Y_pred_mean = np.mean(Y_pred_heads, axis=0).flatten()
            Y_pred_std = np.std(Y_pred_heads, axis=0).flatten()

        print("Time used on inference:", time.time() - start_time)
        if plot:
            plot_uncertainty_regression(self.X_train, self.Y_train, X_pred, Y_pred_mean, Y_pred_std)

    def close_session(self):
        self.sess.close()
