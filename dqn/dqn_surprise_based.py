"""
Modified surprise-based exploration version of DQN from https://github.com/tatsuyaokubo/dqn
"""
import csv
import os
import gym
import random
import numpy as np
import pickle
import tensorflow as tf
from collections import deque

import time
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, Dense
from sklearn.preprocessing import normalize
from keras_autoencoders import build_full_conv_autoencoder_new, plot_images, encode_to_samples, build_deep_predictor, \
    train_autoencoder, build_conv_combo_autoencoder

ENV_NAME = 'Pong-v0'  # Environment name
# ENV_NAME = 'Freeway-v0'  # Environment name
# ENV_NAME = 'MontezumaRevenge-v0'  # Environment name
FRAME_WIDTH = 84  # Resized frame width
FRAME_HEIGHT = 84  # Resized frame height
NUM_EPISODES = 8000  # Number of episodes the agent plays
# TODO might change this to abort on timesteps
STATE_LENGTH = 4  # Number of most recent frames to produce the input to the network
GAMMA = 0.99  # Discount factor
EXPLORATION_STEPS = 1000000  # Number of steps over which the initial value of epsilon is linearly annealed to its final value
INITIAL_EPSILON = 1.0  # Initial value of epsilon in epsilon-greedy
FINAL_EPSILON = 0.1  # Final value of epsilon in epsilon-greedy
INITIAL_REPLAY_SIZE = 50000  # Number of steps to populate the replay memory before training starts
NUM_REPLAY_MEMORY = 300000  # Number of replay memory the agent uses for training
BATCH_SIZE = 32  # Mini batch size
TARGET_UPDATE_INTERVAL = 10000  # The frequency with which the target network is updated
TRAIN_INTERVAL = 4  # The agent selects 4 actions between successive updates
LEARNING_RATE = 0.00025  # Learning rate used by RMSProp
MOMENTUM = 0.95  # Momentum used by RMSProp
MIN_GRAD = 0.01  # Constant added to the squared gradient in the denominator of the RMSProp update
SAVE_INTERVAL = 300000  # The frequency with which the network is saved
NO_OP_STEPS = 30  # Maximum number of "do nothing" actions to be performed by the agent at the start of an episode
LOAD_NETWORK = False
TRAIN = True
SAVE_NETWORK_PATH = 'saved_networks_surprise/' + ENV_NAME
SAVE_SUMMARY_PATH = 'summary/' + ENV_NAME + '_surprise'
NUM_EPISODES_AT_TEST = 30  # Number of episodes the agent plays at test time

UPDATE_DYNAMICS_MODEL = 0.05  # Probability for updating the dynamics model
UPDATE_PREDICTION_MODEL_THRESHOLD = 0.001  # Loss threshold from autoencoder for updating the prediction model
PREDICTOR_LOSS_THRESHOLD = 20.0  # Loss threshold from the prediction network before using surprise-based exploration
Z_SHAPE = 256  # Shape for one single frame


class Agent():
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.epsilon = INITIAL_EPSILON
        self.epsilon_step = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS
        self.t = 0

        # Parameters used for summary
        self.total_reward = 0
        self.total_q_max = 0
        self.total_loss = 0
        self.duration = 0
        self.episode = 0
        self.avg_action_uncertainty = 0
        self.std_action_uncertainty = 0
        self.reward_history = []

        # Create replay memory
        self.replay_memory = deque()

        # Create dynamics model
        self.z_shape = Z_SHAPE
        # self.autoencoder, self.encoder, self.decoder = build_full_conv_autoencoder_new(self.z_shape)
        self.autoencoder, self.encoder, self.decoder = build_conv_combo_autoencoder(self.z_shape)
        self.predictor, self.predictor_loss_function, self.predictor_optimizer, self.keep_prob, self.predictor_x, self.predictor_y = build_deep_predictor(
            self.z_shape * 4, num_actions, self.z_shape)  # x4 for 4 frames concatenated
        self.use_prediction = False

        # Create q network
        self.s, self.q_values, q_network = self.build_network()
        q_network_weights = q_network.trainable_weights

        # Create target network
        self.st, self.target_q_values, target_network = self.build_network()
        target_network_weights = target_network.trainable_weights

        # Define target network update operation
        self.update_target_network = [target_network_weights[i].assign(q_network_weights[i]) for i in
                                      range(len(target_network_weights))]

        # Define loss and gradient update operation
        self.a, self.y, self.loss, self.grads_update = self.build_training_op(q_network_weights)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.InteractiveSession(config=config)
        self.saver = tf.train.Saver(q_network_weights)
        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(SAVE_SUMMARY_PATH, self.sess.graph)

        if not os.path.exists(SAVE_NETWORK_PATH):
            os.makedirs(SAVE_NETWORK_PATH)

        self.sess.run(tf.global_variables_initializer())

        # Load network
        if LOAD_NETWORK:
            self.load_network()

        # Initialize target network
        self.sess.run(self.update_target_network)

    def plot_examples(self):
        state_batch = []
        action_batch = []

        # Sample random minibatch of transition from replay memory
        minibatch = random.sample(self.replay_memory, 2)
        for data in minibatch:
            state_batch.append(data[0])
            action_batch.append(data[1])
        x_batch = np.float32(np.array(state_batch) / 255.0).reshape(8, 84, 84, 1)
        autoencoder_result = self.autoencoder.predict(x_batch, batch_size=BATCH_SIZE)
        print("Actions in plot:", action_batch)
        # x_batch = x_batch.reshape(8, 84, 84, 1)
        # autoencoder_result = autoencoder_result.reshape(8, 84, 84, 1)
        plot_images(x_batch, autoencoder_result, nb_examples=8)

    def build_network(self):
        model = Sequential()
        model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same', activation='relu',
                                input_shape=(STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT)))
        model.add(Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same', activation='relu'))
        model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same', activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))  # TODO use this as the base for the autoencoder?
        model.add(Dense(self.num_actions))

        model.summary()

        s = tf.placeholder(tf.float32, [None, STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT])
        q_values = model(s)

        return s, q_values, model

    def calculate_uncertainty_bonus(self, X_test):
        T = 20
        action_mean = []
        action_std = []
        for a in range(self.num_actions):
            encoded_concatenated_vec = encode_to_samples(self.encoder,
                                                         np.array(X_test).reshape((len(X_test) * 4, 84, 84, 1)),
                                                         [a], self.z_shape, self.num_actions, print_labels=False)
            enumis = []
            for x_test in encoded_concatenated_vec:
                for i in range(0, T):
                    enumis.append(x_test)
            enumis = np.array(enumis)
            prob = self.sess.run(self.predictor, feed_dict={self.predictor_x: enumis, self.keep_prob: 0.9})
            # Y_pred_reshaped = prob.reshape((len(X_test), self.z_shape))
            Y_pred_mean = np.mean(prob, axis=0)
            Y_pred_std = np.std(prob, axis=0)

            action_mean.append(np.mean(Y_pred_mean))
            action_std.append(np.mean(Y_pred_std))

        action_std = np.array(action_std)
        # action_std_softmax = np.exp(action_std - np.max(action_std))
        # action_std_softmax = action_std_softmax / action_std_softmax.sum()
        # action_std_norm = normalize(action_std[:, np.newaxis], norm='l1', axis=0).ravel()

        # print("Softmax:", action_std_softmax, "Pred STD:", action_std)

        return action_std * 2

    def build_training_op(self, q_network_weights):
        a = tf.placeholder(tf.int64, [None])
        y = tf.placeholder(tf.float32, [None])

        # Convert action to one hot vector
        a_one_hot = tf.one_hot(a, self.num_actions, 1.0, 0.0)
        q_value = tf.reduce_sum(tf.multiply(self.q_values, a_one_hot), axis=1)

        # Clip the error, the loss is quadratic when the error is in (-1, 1), and linear outside of that region
        error = tf.abs(y - q_value)
        quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)

        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, momentum=MOMENTUM, epsilon=MIN_GRAD)
        grads_update = optimizer.minimize(loss, var_list=q_network_weights)

        return a, y, loss, grads_update

    def get_initial_state(self, observation, last_observation):
        processed_observation = np.maximum(observation, last_observation)
        processed_observation = np.uint8(resize(rgb2gray(processed_observation), (FRAME_WIDTH, FRAME_HEIGHT)) * 255)
        state = [processed_observation for _ in range(STATE_LENGTH)]
        return np.stack(state, axis=0)

    def get_action(self, state):
        if self.t < INITIAL_REPLAY_SIZE or not self.use_prediction:
            # Random action in beginning
            return random.randrange(self.num_actions)

        if self.epsilon >= random.random():
            # Surprise-based exploration
            converted_state = [np.float32(state / 255.0)]
            q_values_actions = self.q_values.eval(feed_dict={self.s: converted_state})[0]
            uncertainty_values_actions = self.calculate_uncertainty_bonus(converted_state)

            # q_mean = np.mean(q_values_actions)
            q_min = np.min(q_values_actions)
            if q_min < 0:
                q_values_actions = q_values_actions + abs(q_min)
            q_values_actions = normalize(q_values_actions[:, np.newaxis], norm='l1', axis=0).ravel()

            u_min = np.min(uncertainty_values_actions)
            if u_min < 0:
                uncertainty_values_actions = uncertainty_values_actions + u_min
            # linear_transform_const = q_mean-u_mean

            uncertainty_values_actions = normalize(uncertainty_values_actions[:, np.newaxis], norm='l1', axis=0).ravel()
            surprise_bonus = uncertainty_values_actions * pow(self.epsilon, 0.3)  # 0.3 is an exploration constant
            u_mean = np.mean(surprise_bonus)
            u_std = np.std(surprise_bonus)
            self.avg_action_uncertainty = u_mean
            self.std_action_uncertainty = u_std

            final_action_values = q_values_actions + surprise_bonus

            if random.random() < 0.0002:
                print("Q:", q_values_actions)
                print("U:", uncertainty_values_actions)
                print("S:", surprise_bonus)
                print("V:", final_action_values)

            action = np.argmax(final_action_values)
        else:
            # Exploitation
            action = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}))

        # Anneal epsilon linearly over time
        if self.epsilon > FINAL_EPSILON and self.t >= INITIAL_REPLAY_SIZE:
            self.epsilon -= self.epsilon_step

        return action

    def run(self, state, action, reward, terminal, observation):
        next_state = np.append(state[1:, :, :], observation, axis=0)

        # Clip all positive rewards at 1 and all negative rewards at -1, leaving 0 rewards unchanged
        reward = np.clip(reward, -1, 1)

        # Store transition in replay memory
        self.replay_memory.append((state, action, reward, next_state, terminal))
        if len(self.replay_memory) > NUM_REPLAY_MEMORY:
            self.replay_memory.popleft()

        if self.t >= INITIAL_REPLAY_SIZE:
            # Train network
            if self.t % TRAIN_INTERVAL == 0:
                self.train_network()

            # Update target network
            if self.t % TARGET_UPDATE_INTERVAL == 0:
                self.sess.run(self.update_target_network)

            # Save network
            if self.t % SAVE_INTERVAL == 0:
                save_path = self.saver.save(self.sess, SAVE_NETWORK_PATH + '/' + ENV_NAME, global_step=self.t)
                print('Successfully saved: ' + save_path)

        self.total_reward += reward
        self.total_q_max += np.max(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}))
        self.duration += 1

        if terminal:
            # Write summary
            if self.t >= INITIAL_REPLAY_SIZE:
                stats = [self.total_reward, self.total_q_max / float(self.duration),
                         self.duration, self.total_loss / (float(self.duration) / float(TRAIN_INTERVAL)),
                         self.avg_action_uncertainty, self.std_action_uncertainty]
                for i in range(len(stats)):
                    self.sess.run(self.update_ops[i], feed_dict={self.summary_placeholders[i]: float(stats[i])})
                summary_str = self.sess.run(self.summary_op)
                self.summary_writer.add_summary(summary_str, self.episode + 1)

            # Debug
            if self.t < INITIAL_REPLAY_SIZE:
                mode = 'random'
            elif INITIAL_REPLAY_SIZE <= self.t < INITIAL_REPLAY_SIZE + EXPLORATION_STEPS:
                mode = 'explore'
            else:
                mode = 'exploit'
            print(
                'EPISODE: {0:6d} / TIMESTEP: {1:8d} / DURATION: {2:5d} / EPSILON: {3:.5f} / TOTAL_REWARD: {4:3.0f} / AVG_MAX_Q: {5:2.4f} / AVG_LOSS: {6:.5f} / MODE: {7}'.format(
                    self.episode + 1, self.t, self.duration, self.epsilon,
                    self.total_reward, self.total_q_max / float(self.duration),
                    self.total_loss / (float(self.duration) / float(TRAIN_INTERVAL)), mode))
            self.reward_history.append([self.t, self.total_reward])

            self.total_reward = 0
            self.total_q_max = 0
            self.total_loss = 0
            self.duration = 0
            self.episode += 1
            self.avg_action_uncertainty = 0
            self.std_action_uncertainty = 0

        self.t += 1

        return next_state

    def train_network(self):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        terminal_batch = []
        y_batch = []

        # Sample random minibatch of transition from replay memory
        minibatch = random.sample(self.replay_memory, BATCH_SIZE)
        for data in minibatch:
            state_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            next_state_batch.append(data[3])
            terminal_batch.append(data[4])

        # Convert True to 1, False to 0
        terminal_batch = np.array(terminal_batch) + 0

        target_q_values_batch = self.target_q_values.eval(
            feed_dict={self.st: np.float32(np.array(next_state_batch) / 255.0)})
        y_batch = reward_batch + (1 - terminal_batch) * GAMMA * np.max(target_q_values_batch, axis=1)

        x_batch = np.float32(np.array(state_batch) / 255.0)

        loss, _ = self.sess.run([self.loss, self.grads_update], feed_dict={
            self.s: x_batch,
            self.a: action_batch,
            self.y: y_batch
        })

        self.total_loss += loss

        if random.random() < UPDATE_DYNAMICS_MODEL:
            # Update autoencoder
            x_batch = x_batch.reshape((BATCH_SIZE * 4, 84, 84, 1))
            max_loss = train_autoencoder(self.autoencoder, x_batch, None, batch_size=BATCH_SIZE, nb_epoch=1, verbose=0)
            predictor_loss = None
            if max_loss < UPDATE_PREDICTION_MODEL_THRESHOLD:
                # Update prediction model
                predictor_x, predictor_y = encode_to_samples(self.encoder, x_batch, action_batch, self.z_shape,
                                                             self.num_actions, split_inputs=False,
                                                             concat_latent_vectors=True)
                predictor_loss, _ = self.sess.run([self.predictor_loss_function, self.predictor_optimizer],
                                                  feed_dict={self.predictor_x: predictor_x,
                                                             self.predictor_y: predictor_y, self.keep_prob: 0.9})
                if not self.use_prediction and predictor_loss < PREDICTOR_LOSS_THRESHOLD:
                    self.use_prediction = True
                    # prediction_loss = train_predictor(self.predictor, predictor_x, predictor_y, nb_epoch=1,
                    #                                   batch_size=BATCH_SIZE)
            if random.random() < 0.05:
                print("Autoencoder max-loss:", max_loss, "Prediction loss:", predictor_loss)

    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Total Reward/Episode', episode_total_reward)
        # timestep_reward = tf.Variable(0.)
        # tf.summary.scalar(ENV_NAME + '/Total Reward/Timestep', timestep_reward)
        episode_avg_max_q = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Average Max Q/Episode', episode_avg_max_q)
        episode_duration = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Duration/Episode', episode_duration)
        episode_avg_loss = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Average Loss/Episode', episode_avg_loss)
        avg_action_uncertainty = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Average action uncertainty/Episode', avg_action_uncertainty)
        std_action_uncertainty = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Std action uncertainty/Episode', std_action_uncertainty)
        summary_vars = [episode_total_reward, episode_avg_max_q, episode_duration, episode_avg_loss,
                        avg_action_uncertainty, std_action_uncertainty]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

    def load_network(self):
        checkpoint = tf.train.get_checkpoint_state(SAVE_NETWORK_PATH)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print('Successfully loaded: ' + checkpoint.model_checkpoint_path)
        else:
            print('Training new network...')

    def get_action_at_test(self, state):
        if random.random() <= 0.05:
            action = random.randrange(self.num_actions)
        else:
            action = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}))

        self.t += 1

        return action


def preprocess(observation, last_observation):
    processed_observation = np.maximum(observation, last_observation)
    processed_observation = np.uint8(resize(rgb2gray(processed_observation), (FRAME_WIDTH, FRAME_HEIGHT)) * 255)
    return np.reshape(processed_observation, (1, FRAME_WIDTH, FRAME_HEIGHT))


def main():
    env = gym.make(ENV_NAME)
    agent = Agent(num_actions=env.action_space.n)

    if TRAIN:  # Train mode
        training_time_start = time.time()
        for _ in range(NUM_EPISODES):
            terminal = False
            observation = env.reset()
            for _ in range(random.randint(1, NO_OP_STEPS)):
                last_observation = observation
                observation, _, _, _ = env.step(0)  # Do nothing
            state = agent.get_initial_state(observation, last_observation)
            while not terminal:
                last_observation = observation
                action = agent.get_action(state)
                observation, reward, terminal, _ = env.step(action)
                # env.render()
                processed_observation = preprocess(observation, last_observation)
                state = agent.run(state, action, reward, terminal, processed_observation)
        print("Used training time:", time.time() - training_time_start)
        agent.plot_examples()
        pickle.dump(agent.reward_history, open("reward_history.pickle", "wb"))
        with open("reward_history.csv", "wb") as f:
            writer = csv.writer(f)
            writer.writerows(agent.reward_history)
    else:  # Test mode
        # env.monitor.start(ENV_NAME + '-test')
        for _ in range(NUM_EPISODES_AT_TEST):
            terminal = False
            observation = env.reset()
            for _ in range(random.randint(1, NO_OP_STEPS)):
                last_observation = observation
                observation, _, _, _ = env.step(0)  # Do nothing
            state = agent.get_initial_state(observation, last_observation)
            while not terminal:
                last_observation = observation
                action = agent.get_action_at_test(state)
                observation, _, terminal, _ = env.step(action)
                env.render()
                processed_observation = preprocess(observation, last_observation)
                state = np.append(state[1:, :, :], processed_observation, axis=0)
                # env.monitor.close()


if __name__ == '__main__':
    time_start = time.time()
    main()
    print("Used a total of:", time.time() - time_start)
