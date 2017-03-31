import random
import time
from collections import deque
from multiprocessing import Queue
from multiprocessing.sharedctypes import RawArray
from ctypes import c_uint, c_float

from sklearn.preprocessing import normalize

from actor_learner import *
import logging
from matplotlib import pyplot as plt
from emulator_runner import EmulatorRunner
import numpy as np
import custom_logging


class PAACLearner(ActorLearner):
    def __init__(self, network, environment_creator, args):
        super(PAACLearner, self).__init__(network, environment_creator, args)
        self.workers = args.emulator_workers
        actions_header = ','.join(['Action{}'.format(i) for i in range(self.num_actions)])
        logger_config = [{'name': 'environment', 'file_name': 'env_log_0.txt',
                          'header': 'Relative time|Absolute time|Global Time Step|Environment nr|Steps to finish|Reward|[{}]'.format(
                              actions_header)},
                         {'name': 'learning', 'file_name': 'learn_log_0.txt',
                          'header': 'Relative time|Absolute time|Global Time Step|[Advantage]|[R]|[Value]|[Value Discrepancy]|Dynamics_loss|Autoencoder loss|Mean action uncertainty|Std action uncertainty'}]
        self.stats_logger = custom_logging.StatsLogger(logger_config, subfolder=args.debugging_folder)

        self.initial_exploration_constant = 1.
        self.exploration_constant_min = 0.01
        self.exploration_discount = 1. / 6000000.

        # Create replay memory
        self.max_replay_size_autoencoder = 64000
        self.max_replay_size_dynamics = int(
            self.max_replay_size_autoencoder / (self.emulator_counts * self.max_local_steps))
        self.replay_memory_dynamics = deque(maxlen=self.max_replay_size_dynamics)
        self.replay_memory_autoencoder = deque(maxlen=self.max_replay_size_dynamics)
        # self.replay_memory_autoencoder_diff_frames = deque(self.max_replay_size_dynamics)

    @staticmethod
    def choose_next_actions(network, num_actions, states, emulator_counts, session):
        network_output_v, network_output_pi = session.run(
            [network.output_layer_v, network.output_layer_pi, network.encoder_output],
            feed_dict={network.input_ph: states})

        action_indices = PAACLearner.__boltzmann(network_output_pi)

        new_actions = np.eye(num_actions)[action_indices]

        return new_actions, network_output_v, network_output_pi, None, None

    @staticmethod
    def choose_next_actions_surprise(network, num_actions, states, emulator_counts, session, exploration_const):
        autoencoder_states = states[:, :, :, 3].reshape(32, 84, 84, 1)
        # autoencoder_states = np.reshape(states.transpose(0, 3, 1, 2), (emulator_counts * 4, 84, 84, 1))

        network_output_v, network_output_pi, network_latent_var = session.run(
            [network.output_layer_v, network.output_layer_pi, network.encoder_output],
            feed_dict={network.input_ph: states, network.autoencoder_input_ph: autoencoder_states})

        # TODO try to do the next 15 lines in tensorflow on the gpu
        T = 30  # number of stochastic forward passes
        flat_latent_var_four_frames = network_latent_var  # .reshape(emulator_counts, network.latent_shape * 4)
        action_uncertainties = []
        for a in range(num_actions):
            action_repeat = np.repeat([np.eye(num_actions)[a]], T * emulator_counts, axis=0)
            latent_repeat = np.repeat(flat_latent_var_four_frames, T, axis=0)
            feed_dict = {network.keep_prob: .9, network.dynamics_input: latent_repeat,
                         network.action_input: action_repeat}
            transition_predictions = session.run(network.latent_prediction, feed_dict=feed_dict)
            transition_predictions = transition_predictions.reshape(emulator_counts, T, network.latent_shape)
            action_uncertainties.append(np.mean(np.std(transition_predictions, axis=1) * 2, axis=1))

            """
            TODO
            GPU implementation of uncertainty calculation:

            ac_rep = tf.reshape(tf.tile(tf.eye(num_actions), T*emulator_counts), [-1, 1])
            lat_rep = tf.reshape(tf.tile(flat_latent_var_four_frames, T*num_actions), [-1, 1])
            lat_ac_concat = tf.concat((ac_rep, lat_rep), axis=1)
            transition_predictions = tf.reshape(self.latent_prediction, [emulator_counts, T, network.latent_shape * 4])
            transition_predictions_mean, transition_predictions_var = tf.nn.moments(transition_predictions, 1)
            action_uncertainties = tf.reduce_mean(tf.multiply(tf.pow(tr, 0.5), 2), 1)
            norm_action_uncertainties = tf.nn.softmax(action_uncertainties)
            mean_norm_action_uncertainties = tf.reduce_mean(norm_action_uncertainties)
            delta_action_uncertainties = tf.subtract(norm_action_uncertainties, mean_norm_action_uncertainties)

            ...

            feed_dict = {network.keep_prob: .9, network.latent_var_four_frames_input: flat_latent_var_four_frames, network.stochastic_feedforwards: T}
            action_uncertainties = session.run(network.action_uncertainty, feed_dict=feed_dict)
            transition_predictions = transition_predictions.reshape(emulator_counts, T, network.latent_shape * 4)
            action_uncertainties.append(np.mean(np.std(transition_predictions, axis=1) * 2, axis=1))
            """

        action_uncertainties = normalize(np.array(action_uncertainties).transpose(), norm='l1', axis=1)
        action_uncertainties = (action_uncertainties - action_uncertainties.mean()) * exploration_const

        if action_uncertainties.std() > 0.:
            network_output_pi_w_surprise = np.clip(np.add(network_output_pi, action_uncertainties), 0., 1.)

            # Probability matching
            action_indices = PAACLearner.__boltzmann(normalize(network_output_pi_w_surprise, norm='l1', axis=1))

            if random.random() < 0.0001:
                print("output_pi:", network_output_pi[0], "\nUncertainties", action_uncertainties[0],
                      "\noutput_pi_surprise:", network_output_pi_w_surprise[0])
        else:
            # Regular boltzmann if there is no uncertainty
            action_indices = PAACLearner.__boltzmann(network_output_pi)

        # UCB
        # network_output_pi_w_surprise = np.add(network_output_pi, action_uncertainties)
        # action_indices = PAACLearner.__ucb1(network_output_pi_w_surprise)

        new_actions = np.eye(num_actions)[action_indices]

        return new_actions, network_output_v, network_output_pi, flat_latent_var_four_frames, action_uncertainties

    def __choose_next_actions(self, states):
        # TODO check if network is using surprise exploration architecture
        return PAACLearner.choose_next_actions_surprise(self.network, self.num_actions, states, self.emulator_counts,
                                                        self.session, self.__get_exploration_const())

    def __get_exploration_const(self):
        return max(self.exploration_constant_min,
                   self.initial_exploration_constant - (self.global_step * self.exploration_discount))

    @staticmethod
    def __boltzmann(probs):
        """
        Sample an action from an action probability distribution output by
        the policy network.
        """
        # Subtract a tiny value from probabilities in order to avoid
        # "ValueError: sum(pvals[:-1]) > 1.0" in numpy.multinomial
        probs = probs - np.finfo(np.float32).epsneg

        action_indexes = [int(np.nonzero(np.random.multinomial(1, p))[0]) for p in probs]
        return action_indexes

    @staticmethod
    def __ucb1(probs):
        """
        Select the maximum action from a probability distribution.
        """
        return np.argmax(probs, axis=1)

    def _get_shared(self, array, dtype=c_float):
        """
        Returns a RawArray backed numpy array that can be shared between processes.
        :param array: the array to be shared
        :param dtype: the RawArray dtype to use
        :return: the RawArray backed numpy array
        """

        shape = array.shape
        shared = RawArray(dtype, array.reshape(-1))
        return np.frombuffer(shared, dtype).reshape(shape)

    def train(self):
        """
        Main actor learner loop for parallel advantage actor critic learning.
        """

        self.global_step = self.init_network()

        logging.debug("Starting training at Step {}".format(self.global_step))
        counter = 0

        global_step_start = self.global_step

        total_rewards = []

        # state, reward, episode_over, action
        shared_states = self._get_shared(np.asarray([emulator.get_initial_state() for emulator in self.emulators],
                                                    dtype=c_uint), dtype=c_uint)
        shared_rewards = self._get_shared(np.zeros(self.emulator_counts))
        shared_episode_over = self._get_shared(np.asarray([False] * self.emulator_counts))
        shared_actions = self._get_shared(np.zeros((self.emulator_counts, self.num_actions)))

        queues = [Queue() for _ in range(self.workers)]
        barrier = Queue()

        runners = [EmulatorRunner(i, emulators, variables, queues[i], barrier) for i, (emulators, variables) in
                   enumerate(zip(np.split(self.emulators, self.workers), zip(np.split(shared_states, self.workers),
                                                                             np.split(shared_rewards, self.workers),
                                                                             np.split(shared_episode_over,
                                                                                      self.workers),
                                                                             np.split(shared_actions, self.workers))))]

        actions_sum = np.zeros((self.emulator_counts, self.num_actions))

        for r in runners:
            r.start()

        emulator_steps = [0] * self.emulator_counts

        total_episode_rewards = self.emulator_counts * [0]

        y_batch = np.zeros((self.max_local_steps, self.emulator_counts))
        adv_batch = np.zeros((self.max_local_steps, self.emulator_counts))
        rewards = np.zeros((self.max_local_steps, self.emulator_counts))
        states = np.zeros([self.max_local_steps] + list(shared_states.shape), dtype=np.uint8)
        latent_vars = np.zeros((self.max_local_steps, self.emulator_counts, self.network.latent_shape))
        actions = np.zeros((self.max_local_steps, self.emulator_counts, self.num_actions))
        values = np.zeros((self.max_local_steps, self.emulator_counts))
        episodes_over_masks = np.zeros((self.max_local_steps, self.emulator_counts))

        start_time = time.time()

        while self.global_step < self.max_global_steps:

            loop_start_time = time.time()

            mean_action_uncertainty = 0.
            std_action_uncertainty = 0.
            max_local_steps = self.max_local_steps
            for t in range(max_local_steps):
                next_actions, readouts_v_t, readouts_pi_t, latent_var, action_uncertainty = self.__choose_next_actions(
                    shared_states)
                actions_sum += next_actions

                mean_action_uncertainty = action_uncertainty.mean()
                std_action_uncertainty = action_uncertainty.std()

                for z in range(next_actions.shape[0]):
                    shared_actions[z] = next_actions[z]

                actions[t] = next_actions
                values[t] = readouts_v_t
                states[t] = shared_states
                latent_vars[t] = latent_var

                # Start updating all environments with next_actions
                for queue in queues:
                    queue.put(True)
                for wd in range(self.workers):
                    barrier.get()
                # Done updating all environments, have new states, rewards and is_over

                episodes_over_masks[t] = 1.0 - shared_episode_over.astype(np.float32)

                for e, (actual_reward, episode_over) in enumerate(zip(shared_rewards, shared_episode_over)):
                    total_episode_rewards[e] += actual_reward
                    actual_reward = self.rescale_reward(actual_reward)
                    rewards[t, e] = actual_reward

                    emulator_steps[e] += 1
                    self.global_step += 1
                    if episode_over:
                        total_rewards.append(total_episode_rewards[e])
                        self.stats_logger.log('environment', self.global_step, e, emulator_steps[e],
                                              total_episode_rewards[e], actions_sum[e] / emulator_steps[e])
                        total_episode_rewards[e] = 0
                        emulator_steps[e] = 0
                        actions_sum[e] = np.zeros(self.num_actions)

            Rs = self.session.run(self.network.output_layer_v, feed_dict={self.network.input_ph: shared_states})

            Rs_stats = self.stats_logger.get_stats_for_array(Rs)

            n_Rs = np.copy(Rs)

            for t in reversed(range(max_local_steps)):
                n_Rs = rewards[t] + self.gamma * n_Rs * episodes_over_masks[t]
                y_batch[t] = np.copy(n_Rs)
                adv_batch[t] = n_Rs - values[t]

            flat_states = states.reshape([self.max_local_steps * self.emulator_counts] + list(shared_states.shape)[1:])
            flat_y_batch = y_batch.reshape(-1)
            flat_adv_batch = adv_batch.reshape(-1)
            flat_actions = actions.reshape(max_local_steps * self.emulator_counts, self.num_actions)

            # states shape (t_max, emulator_count, 84, 84, 4)
            trans_state = states.transpose(1, 0, 2, 3, 4)
            flat_autoencoder_prev_states = trans_state[:, 1:, :, :, 1].reshape(32 * 4, 84, 84, 1)
            flat_autoencoder_states = trans_state[:, 1:, :, :, 2].reshape(32 * 4, 84, 84, 1)
            flat_autoencoder_next_states = trans_state[:, 1:, :, :, 3].reshape(32 * 4, 84, 84, 1)
            flat_autoencoder_actions = actions.transpose(1, 0, 2)[:, :-1, :].reshape(32 * 4, self.num_actions)
            flat_autoencoder_diff_frames = flat_autoencoder_states - flat_autoencoder_next_states

            # Store transition data in replay memory (prev, current, next, action, diff)
            self.replay_memory_autoencoder.append(
                (flat_autoencoder_prev_states, flat_autoencoder_states, flat_autoencoder_next_states,
                 flat_autoencoder_actions, flat_autoencoder_diff_frames))
            # self.replay_memory_dynamics.append((dynamics_input, dynamics_target))

            # Training the regular RL CNN
            lr = self.get_lr()
            feed_dict = {self.network.input_ph: flat_states,
                         self.network.critic_target_ph: flat_y_batch,
                         self.network.selected_action_ph: flat_actions,
                         self.network.adv_actor_ph: flat_adv_batch,
                         self.learning_rate: lr}
            _, value_discrepancy = self.session.run([self.train_step, self.network.value_discrepancy],
                                                    feed_dict=feed_dict)

            # Training the dynamics model
            if counter % (1024 / self.emulator_counts) == 0:
                autoencoder_loss, dynamics_loss = self.train_dynamics_model()

            counter += 1
            self.stats_logger.log('learning', self.global_step,
                                  self.stats_logger.get_stats_for_array(flat_adv_batch),
                                  self.stats_logger.get_stats_for_array(flat_y_batch), Rs_stats,
                                  self.stats_logger.get_stats_for_array(value_discrepancy), dynamics_loss,
                                  autoencoder_loss, mean_action_uncertainty, std_action_uncertainty)

            if counter % (2048 / self.emulator_counts) == 0:
                curr_time = time.time()
                global_steps = self.global_step
                last_ten = 0.0 if len(total_rewards) < 1 else np.mean(total_rewards[-10:])
                logging.info(
                    "Ran {} steps, at {} steps/s ({} steps/s avg), last 10 rewards avg {}, dynamics loss {}, autoencoder loss {}, action uncertainty std {}"
                        .format(global_steps,
                                self.max_local_steps * self.emulator_counts / (curr_time - loop_start_time),
                                (global_steps - global_step_start) / (curr_time - start_time),
                                last_ten, dynamics_loss, autoencoder_loss, std_action_uncertainty))

            if counter % (20480 / self.emulator_counts) == 0:
                self.do_plotting()

            self.save_vars()

        for queue in queues:
            queue.put(None)
        self.save_vars(True)

        self.do_plotting()

    def train_dynamics_model(self):
        # Optimize autoencoder
        num_epochs = 100
        autoencoder_loss = 0.
        dynamics_loss = 0.
        # TODO fix this schedule such that either the autoencoder is trained or the dynamics prediction is trained
        if 50000 < self.global_step:
            for i in range(num_epochs):
                sample = random.sample(self.replay_memory_autoencoder, 1)[0]
                idx_batch = np.random.randint(len(sample[0]), size=32)  # Find random batch
                minibatch_prev_frames = sample[0][idx_batch]
                minibatch_curr_frames = sample[1][idx_batch]
                minibatch_next_frames = sample[2][idx_batch]
                minibatch_actions = sample[3][idx_batch]
                minibatch_focus = sample[4][idx_batch]
                feed_dict = {
                    self.network.autoencoder_input_ph: minibatch_curr_frames,
                    self.network.autoencoder_movement_focus_input_ph: minibatch_focus}
                _, autoencoder_loss, latent_vars = self.session.run(
                    [self.network.autoencoder_optimizer, self.network.autoencoder_loss, self.network.encoder_output],
                    feed_dict=feed_dict)

                feed_dict = {self.network.autoencoder_input_ph: minibatch_next_frames}
                next_latent_vars = self.session.run(self.network.encoder_output, feed_dict=feed_dict)

                feed_dict = {self.network.dynamics_input: latent_vars,
                             self.network.action_input: minibatch_actions,
                             self.network.dynamics_latent_target: next_latent_vars,
                             self.network.keep_prob: .9}
                _, dynamics_loss = self.session.run([self.network.dynamics_optimizer, self.network.dynamics_loss],
                                                    feed_dict=feed_dict)

        return autoencoder_loss, dynamics_loss

    def do_plotting(self, save_imgs=True):
        # Plot autoencoder
        n_imgs = 4
        sample = random.sample(self.replay_memory_autoencoder, 1)[0]
        idx_batch = np.arange(n_imgs)
        minibatch_prev_frames = sample[0][idx_batch]
        minibatch_curr_frames = sample[1][idx_batch]
        minibatch_next_frames = sample[2][idx_batch]
        minibatch_actions = sample[3][idx_batch]

        feed_dict = {self.network.autoencoder_input_ph: minibatch_curr_frames}
        autoencoder_predict, latent_vars = self.session.run(
            [self.network.autoencoder_output, self.network.encoder_output], feed_dict=feed_dict)
        plot_autoencoder_examples([minibatch_curr_frames, autoencoder_predict], nb_examples=n_imgs, show_plot=not save_imgs,
                                  save_fig=save_imgs,
                                  save_path=self.debugging_folder + '/autoencoder_imgs/' + str(self.global_step))

        # Plot transition prediction
        feed_dict = {self.network.autoencoder_input_ph: minibatch_next_frames}
        dynamics_target = self.session.run(self.network.encoder_output, feed_dict=feed_dict)

        feed_dict = {self.network.dynamics_input: latent_vars, self.network.action_input: minibatch_actions,
                     self.network.keep_prob: 1.}
        predicted_vars = self.session.run(self.network.latent_prediction, feed_dict=feed_dict)
        predicted_vars = np.add(latent_vars, predicted_vars)

        feed_dict = {self.network.decoder_input: predicted_vars}
        predicted_images = self.session.run(self.network.decoder_output, feed_dict=feed_dict)

        plot_autoencoder_examples([minibatch_curr_frames, minibatch_next_frames, predicted_images], nb_examples=n_imgs,
                                  show_plot=not save_imgs,
                                  save_fig=save_imgs,
                                  save_path=self.debugging_folder + '/dynamics_imgs/' + str(self.global_step))

        possible_actions = np.eye(self.num_actions)
        repeated_latent_var = np.repeat(latent_vars, self.num_actions, axis=0)[:self.num_actions]
        feed_dict = {self.network.dynamics_input: repeated_latent_var, self.network.action_input: possible_actions,
                     self.network.keep_prob: 1.}
        predicted_vars = self.session.run(self.network.latent_prediction, feed_dict=feed_dict)
        dynamics_target = np.repeat(minibatch_next_frames[0], self.num_actions, axis=2).transpose(2, 0, 1)
        predicted_vars = np.add(latent_vars, predicted_vars)

        feed_dict = {self.network.decoder_input: predicted_vars}
        predicted_images = self.session.run(self.network.decoder_output, feed_dict=feed_dict)
        org_img_repeat = np.repeat(minibatch_curr_frames[0], self.num_actions, axis=2).transpose(2, 0, 1)
        plot_autoencoder_examples([org_img_repeat, predicted_images], nb_examples=self.num_actions,
                                  show_plot=not save_imgs, save_fig=save_imgs,
                                  save_path=self.debugging_folder + '/dynamics_imgs/' + str(
                                      self.global_step) + '_action_variance')


def plot_autoencoder_examples(img_array, nb_examples=10, img_width=84, img_height=84, show_plot=True, save_fig=False,
                              save_path=''):
    n = nb_examples

    rows = len(img_array)

    for i in range(rows):
        img_array[i] = np.reshape(img_array[i], (n, img_width, img_height))

    plt.figure(figsize=(nb_examples * rows, 4))
    plt.gray()
    for i in range(n):
        for j in range(rows):
            ax = plt.subplot(rows, n, i + 1 + (j * n))
            plt.imshow(img_array[j][i])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    if save_fig:
        plt.savefig(save_path + '.png', bbox_inches='tight')
        if not show_plot:
            plt.close()
    if show_plot:
        plt.show()
