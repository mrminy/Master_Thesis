import random
import time
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
        autoencoder_states = np.reshape(states.transpose(0, 3, 1, 2), (emulator_counts * 4, 84, 84, 1))

        network_output_v, network_output_pi, network_latent_var = session.run(
            [network.output_layer_v, network.output_layer_pi, network.encoder_output],
            feed_dict={network.input_ph: states, network.autoencoder_input_ph: autoencoder_states})

        # TODO try to do the next 15 lines in tensorflow on the gpu
        T = 30  # number of stochastic forward passes
        flat_latent_var_four_frames = network_latent_var.reshape(emulator_counts, network.latent_shape * 4)
        action_uncertainties = []
        for a in range(num_actions):
            action_repeat = np.repeat([np.eye(num_actions)[a]], T * emulator_counts, axis=0)
            latent_repeat = np.repeat(flat_latent_var_four_frames, T, axis=0)
            flat_concatenated_latent_actions = np.concatenate((latent_repeat, action_repeat), axis=1)
            feed_dict = {network.keep_prob: .9, network.dynamics_input: flat_concatenated_latent_actions}
            transition_predictions = session.run(network.latent_prediction, feed_dict=feed_dict)
            transition_predictions = transition_predictions.reshape(emulator_counts, T, network.latent_shape)
            action_uncertainties.append(np.mean(np.std(transition_predictions, axis=1) * 2, axis=1))

            """
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

        # if action_uncertainties.std() > 0.:
        #     network_output_pi_w_surprise = np.clip(np.add(network_output_pi, action_uncertainties), 0., 1.)
        #
            # Probability matching
            # action_indices = PAACLearner.__boltzmann(normalize(network_output_pi_w_surprise, norm='l1', axis=1))
            #
            # if random.random() < 0.0007:
            #     print("output_pi:", network_output_pi[0], "\nUncertainties", action_uncertainties[0],
            #           "\noutput_pi_surprise:", network_output_pi_w_surprise[0])
        # else:
        #   # Regular boltzmann if there is no uncertainty
            # action_indices = PAACLearner.__boltzmann(network_output_pi)

        # UCB TODO try this
        network_output_pi_w_surprise = np.add(network_output_pi, action_uncertainties)
        action_indices = PAACLearner.__ucb1(network_output_pi_w_surprise)

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
        return np.argmax(probs)

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
        latent_vars = np.zeros((self.max_local_steps, self.emulator_counts, self.network.latent_shape * 4))
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

            Rs = self.session.run(
                self.network.output_layer_v,
                feed_dict={self.network.input_ph: shared_states})

            Rs_stats = self.stats_logger.get_stats_for_array(Rs)

            n_Rs = np.copy(Rs)

            for t in reversed(range(max_local_steps)):
                n_Rs = rewards[t] + self.gamma * n_Rs * episodes_over_masks[t]
                y_batch[t] = np.copy(n_Rs)
                adv_batch[t] = n_Rs - values[t]

            flat_states = states.reshape([self.max_local_steps * self.emulator_counts] + list(shared_states.shape)[1:])
            flat_autoencoder_states = states.transpose(1, 0, 4, 2, 3).reshape(self.emulator_counts,
                                                                              self.max_local_steps * 4, 84, 84, 1)
            flat_y_batch = y_batch.reshape(-1)
            flat_adv_batch = adv_batch.reshape(-1)
            flat_actions = actions.reshape(max_local_steps * self.emulator_counts, self.num_actions)
            flat_latent_vars_four_frames = latent_vars.reshape(max_local_steps * self.emulator_counts,
                                                               self.network.latent_shape * 4)
            flat_concatenated_latent_actions = np.concatenate((flat_latent_vars_four_frames, flat_actions),
                                                              axis=1).reshape(self.max_local_steps,
                                                                              self.emulator_counts,
                                                                              self.network.latent_shape * 4 + self.num_actions)

            lr = self.get_lr()
            feed_dict = {self.network.input_ph: flat_states,
                         self.network.critic_target_ph: flat_y_batch,
                         self.network.selected_action_ph: flat_actions,
                         self.network.adv_actor_ph: flat_adv_batch,
                         self.learning_rate: lr}
            _, value_discrepancy = self.session.run([self.train_step, self.network.value_discrepancy],
                                                    feed_dict=feed_dict)

            autoencoder_loss, dynamics_loss = self.train_dynamics_model(flat_autoencoder_states,
                                                                        flat_concatenated_latent_actions, latent_vars)

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

            if False and counter % (20480 / self.emulator_counts) == 0:
                self.do_plotting(flat_autoencoder_states, flat_actions)

            self.save_vars()

        for queue in queues:
            queue.put(None)
        self.save_vars(True)

        if flat_autoencoder_states is not None and flat_actions is not None:
            self.do_plotting(flat_autoencoder_states, flat_actions)

    def train_dynamics_model(self, flat_autoencoder_states, flat_concatenated_latent_actions, latent_vars_four_frames):
        # Optimize autoencoder
        autoencoder_loss = 0.
        flat_autoencoder_states_re = flat_autoencoder_states.reshape(self.emulator_counts * self.max_local_steps * 4,
                                                                     84, 84, 1)
        if self.global_step < 250000:
            for _ in range(4):
                feed_dict = {self.network.autoencoder_input_ph: flat_autoencoder_states_re[
                    np.random.randint(flat_autoencoder_states_re.shape[0], size=32)]}
                _, autoencoder_loss = self.session.run(
                    [self.network.autoencoder_optimizer, self.network.autoencoder_loss], feed_dict=feed_dict)
        else:
            feed_dict = {self.network.autoencoder_input_ph: flat_autoencoder_states_re[
                np.random.randint(flat_autoencoder_states_re.shape[0], size=32)]}
            _, autoencoder_loss = self.session.run(
                [self.network.autoencoder_optimizer, self.network.autoencoder_loss],
                feed_dict=feed_dict)

        # Wait some timesteps before starting training the prediction network
        dynamics_loss = 0.
        if autoencoder_loss <= 0.0002:  # or self.global_step >= 1000000
            dynamics_input = flat_concatenated_latent_actions[:-1].reshape(
                (self.max_local_steps - 1) * self.emulator_counts, self.num_actions + self.network.latent_shape * 4)

            dynamics_target = latent_vars_four_frames.reshape(self.max_local_steps, self.emulator_counts, 4,
                                                              self.network.latent_shape)[1:, :, -1].reshape(
                (self.max_local_steps - 1) * self.emulator_counts, self.network.latent_shape)

            for i in range(4):
                idx_batch = np.random.randint(len(dynamics_input), size=32)  # Find random batch
                feed_dict = {self.network.dynamics_input: dynamics_input[idx_batch],
                             self.network.dynamics_latent_target: dynamics_target[idx_batch],
                             self.network.keep_prob: .9}
                _, dynamics_loss = self.session.run([self.network.dynamics_optimizer, self.network.dynamics_loss],
                                                    feed_dict=feed_dict)

        return autoencoder_loss, dynamics_loss

    def do_plotting(self, flat_autoencoder_states, flat_actions):
        # Plot autoencoder
        n_imgs = 20
        org_imgs = flat_autoencoder_states[0][:n_imgs]
        feed_dict = {self.network.autoencoder_input_ph: org_imgs}
        autoencoder_predict = self.session.run(self.network.autoencoder_output, feed_dict=feed_dict)
        plot_autoencoder_examples(org_imgs, autoencoder_predict, nb_examples=n_imgs)

        # Plot decoder
        # org_latent_vars = [latent_vars[0][0], latent_vars[0][1]]
        # feed_dict = {self.network.decoder_input: org_latent_vars}
        # decoder_predict = self.session.run(self.network.decoder_output, feed_dict=feed_dict)
        # plot_autoencoder_examples(autoencoder_predict[0:2], decoder_predict, nb_examples=2)

        # Plot transition prediction
        n_imgs = 4
        org_actions = flat_actions.reshape(self.emulator_counts, self.max_local_steps, self.num_actions)[0,
                      :self.max_local_steps]

        feed_dict = {self.network.autoencoder_input_ph: org_imgs}
        encoded_vars = self.session.run(self.network.encoder_output, feed_dict=feed_dict)

        flat_latent_vars_four_frames = encoded_vars.reshape(len(org_actions), self.network.latent_shape * 4)
        flat_concatenated_latent_actions = np.concatenate((flat_latent_vars_four_frames, org_actions), axis=1)

        feed_dict = {self.network.dynamics_input: flat_concatenated_latent_actions, self.network.keep_prob: 1.}
        predicted_vars = self.session.run(self.network.latent_prediction, feed_dict=feed_dict)

        feed_dict = {self.network.decoder_input: predicted_vars}
        predicted_images = self.session.run(self.network.decoder_output, feed_dict=feed_dict)

        next_images = org_imgs[4::4]
        plot_autoencoder_examples(next_images, predicted_images[:-1], nb_examples=n_imgs)


def plot_autoencoder_examples(org_images, predicted_images, nb_examples=10):
    n = nb_examples

    org_images = np.reshape(org_images, (n, 84, 84))
    predicted_images = np.reshape(predicted_images, (n, 84, 84))

    plt.figure(figsize=(nb_examples * 2, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(org_images[i])  # .reshape(84, 84))
        # plt.imshow(X_test[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(predicted_images[i])  # .reshape(84, 84))
        # plt.imshow(decoded_imgs[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
