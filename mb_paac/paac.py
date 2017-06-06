"""
Copyright [2017] [Alfredo Clemente]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

--------------------------------------------------------------
Motification
- Converted PAAC to MB-PAAC (model-based). This dynamically trains the deep dynamics model with the policy- and value-network.
- Added functions for extracting the different kinds of intrinsic reward bonuses from the deep dynamics model.
- Added functionality for plotting different reconstructions and predictions of the deep dynamics model.
"""

import random
import time
from collections import deque
from multiprocessing import Queue
from multiprocessing.sharedctypes import RawArray
from ctypes import c_uint, c_float

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
                          'header': 'Relative time|Absolute time|Global Time Step|[Advantage]|[R]|[Value]|[Value Discrepancy]|Dynamics_loss|Autoencoder loss|Mean action uncertainty|Std action uncertainty|Avg reward bonus'}]
        self.stats_logger = custom_logging.StatsLogger(logger_config, subfolder=args.debugging_folder)

        self.enable_plotting = args.enable_plotting

        self.static_ae = args.static_ae

        self.initial_exploration_constant = args.initial_exploration_const
        self.exploration_constant_min = args.final_exploration_const
        self.exploration_discount = 1. / args.end_exploration_discount

        # Create replay memory
        self.max_replay_size = args.replay_size
        self.replay_memory_dynamics_model = deque(maxlen=int(self.max_replay_size / self.emulator_counts))

        # Average autoencoder loss and dynamics loss
        self.autoencoder_loss_mean = 1.
        self.autoencoder_loss_max = 1.
        self.autoencoder_loss_std = 0.
        self.dynamics_loss_mean = 1.
        self.dynamics_loss_max = 1.

        self.min_max_value = 0.1  # intrinsic rewards are clipped +- min_max_value

        # Choose type of intrinsic reward. Default is surprise by MC dropout
        self.update_reward_bonus = self.update_reward_bonus_surprise
        if args.bonus_type == 'AL':
            self.update_reward_bonus = self.update_reward_bonus_ae_loss
        elif args.bonus_type == 'DL':
            self.update_reward_bonus = self.update_reward_bonus_dynamics_loss
        elif args.bonus_type == 'BDU':
            self.update_reward_bonus = self.update_reward_bonus_surprise_boot_and_mc

    @staticmethod
    def choose_next_actions(network, num_actions, states, session):
        network_output_v, network_output_pi = session.run([network.output_layer_v, network.output_layer_pi],
                                                          feed_dict={network.input_ph: states})

        action_indices = PAACLearner.__boltzmann(network_output_pi)

        new_actions = np.eye(num_actions)[action_indices]

        return new_actions, network_output_v, network_output_pi

    def __choose_next_actions(self, states):
        return self.choose_next_actions(self.network, self.num_actions, states, self.session)

    def __get_exploration_const(self):
        return max(self.exploration_constant_min,
                   self.initial_exploration_constant - (self.global_step * self.exploration_discount))

    @staticmethod
    def __boltzmann(probs):
        """
        Sample an action from an action probability distribution output by the policy network.
        """
        # Subtract a tiny value from probabilities in order to avoid
        # "ValueError: sum(pvals[:-1]) > 1.0" in numpy.multinomial
        probs = probs - np.finfo(np.float32).epsneg

        action_indexes = [int(np.nonzero(np.random.multinomial(1, p))[0]) for p in probs]
        return action_indexes

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
        actions = np.zeros((self.max_local_steps, self.emulator_counts, self.num_actions))
        values = np.zeros((self.max_local_steps, self.emulator_counts))
        episodes_over_masks = np.zeros((self.max_local_steps, self.emulator_counts))
        intrinsic_bonus = np.zeros((self.max_local_steps, self.emulator_counts))

        start_time = time.time()

        while self.global_step < self.max_global_steps:

            loop_start_time = time.time()

            max_local_steps = self.max_local_steps
            for t in range(max_local_steps):
                next_actions, readouts_v_t, readouts_pi_t = self.__choose_next_actions(shared_states)
                actions_sum += next_actions

                for z in range(next_actions.shape[0]):
                    shared_actions[z] = next_actions[z]

                actions[t] = next_actions
                values[t] = readouts_v_t
                states[t] = shared_states

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

            intrinsic_bonus = self.update_reward_bonus(states, actions, intrinsic_bonus)
            avg_reward_bonus = np.mean(intrinsic_bonus)

            # Add intrinsic reward bonuses to the rewards
            for t in reversed(range(max_local_steps)):
                n_Rs = rewards[t] + intrinsic_bonus[t] + self.gamma * n_Rs * episodes_over_masks[t]
                y_batch[t] = np.copy(n_Rs)
                adv_batch[t] = n_Rs - values[t]

            flat_states = states.reshape([self.max_local_steps * self.emulator_counts] + list(shared_states.shape)[1:])
            flat_y_batch = y_batch.reshape(-1)
            flat_adv_batch = adv_batch.reshape(-1)
            flat_actions = actions.reshape(max_local_steps * self.emulator_counts, self.num_actions)

            single_frame_shape = (self.emulator_counts * (self.max_local_steps - 1), 84, 84, 1)
            trans_state = states.transpose(1, 0, 2, 3, 4)
            flat_autoencoder_prev_states = trans_state[:, 1:, :, :, -3].reshape(single_frame_shape)
            flat_autoencoder_states = trans_state[:, 1:, :, :, -2].reshape(single_frame_shape)
            flat_autoencoder_next_states = trans_state[:, 1:, :, :, -1].reshape(single_frame_shape)
            flat_autoencoder_actions = actions.transpose(1, 0, 2)[:, :-1, :].reshape(
                self.emulator_counts * (self.max_local_steps - 1), self.num_actions)

            # Store transition data in replay memory as the tuple (prev, current, next, action)
            self.replay_memory_dynamics_model.append(
                (flat_autoencoder_prev_states, flat_autoencoder_states, flat_autoencoder_next_states,
                 flat_autoencoder_actions))

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
            if counter % (1024 / self.emulator_counts) == 0 and self.global_step >= self.max_replay_size:
                autoencoder_loss_arr, dynamics_loss_arr = self.train_dynamics_model(num_epochs=100)
                self.autoencoder_loss_mean = np.mean(autoencoder_loss_arr)
                self.autoencoder_loss_std = np.std(autoencoder_loss_arr)
                self.autoencoder_loss_max = np.max(autoencoder_loss_arr)
                self.dynamics_loss_mean = np.mean(dynamics_loss_arr)
                self.dynamics_loss_max = np.max(dynamics_loss_arr)

            counter += 1
            self.stats_logger.log('learning', self.global_step,
                                  self.stats_logger.get_stats_for_array(flat_adv_batch),
                                  self.stats_logger.get_stats_for_array(flat_y_batch), Rs_stats,
                                  self.stats_logger.get_stats_for_array(value_discrepancy), self.dynamics_loss_mean,
                                  self.autoencoder_loss_mean, 0., 0., avg_reward_bonus)

            if counter % (2048 / self.emulator_counts) == 0:
                curr_time = time.time()
                global_steps = self.global_step
                last_ten = 0.0 if len(total_rewards) < 1 else np.mean(total_rewards[-10:])
                logging.info(
                    "Ran {} steps, at {} steps/s ({} steps/s avg), last 10 rewards avg {}, dynamics loss {}, autoencoder loss {}, avg reward bonus {}"
                        .format(global_steps,
                                self.max_local_steps * self.emulator_counts / (curr_time - loop_start_time),
                                (global_steps - global_step_start) / (curr_time - start_time),
                                last_ten, self.dynamics_loss_mean, self.autoencoder_loss_mean, avg_reward_bonus))

            # Save plots for dynamics model
            if self.enable_plotting and counter % (40960 / self.emulator_counts) == 0:
                self.plot_dynamics_model()

            self.save_vars()

        for queue in queues:
            queue.put(None)
        self.save_vars(True)

        if self.enable_plotting:
            self.plot_dynamics_model()

    def update_reward_bonus_ae_loss(self, states, actions, intrinsic_reward):
        """
        Updates the intrinsic reward bonus with respect to the autoencoder loss.
        :param states: observed states
        :param actions: selected actions (not needed for this reward bonus)
        :param intrinsic_reward: the intrinsic rewards to update
        :return: the updated intrinsic reward bonuses
        """
        for t in range(self.max_local_steps):
            autoencoder_states_curr = states[t, :, :, :, -2].reshape(self.emulator_counts, 84, 84, 1)
            autoencoder_states_next = states[t, :, :, :, -1].reshape(self.emulator_counts, 84, 84, 1)

            feed_dict = {self.network.autoencoder_input_ph: autoencoder_states_curr,
                         self.network.autoencoder_input_next_ph: autoencoder_states_next}
            ae_loss = self.session.run(self.network.emulator_reconstruction_loss, feed_dict=feed_dict)

            ae_loss_bonus = np.divide(ae_loss, self.autoencoder_loss_max)

            intrinsic_reward[t] = ae_loss_bonus

        intrinsic_reward = np.multiply(intrinsic_reward, self.__get_exploration_const())
        intrinsic_reward = np.clip(intrinsic_reward, -self.min_max_value, self.min_max_value)
        return intrinsic_reward

    def update_reward_bonus_dynamics_loss(self, states, actions, intrinsic_reward):
        """
        Updates the intrinsic reward bonus with respect to the dynamics prediction error.
        :param states: observed states
        :param actions: selected actions
        :param intrinsic_reward: the intrinsic rewards to update
        :return: the updated intrinsic reward bonuses
        """
        for t in range(self.max_local_steps):
            autoencoder_states_prev = states[t, :, :, :, -3].reshape(self.emulator_counts, 84, 84, 1)
            autoencoder_states_curr = states[t, :, :, :, -2].reshape(self.emulator_counts, 84, 84, 1)
            autoencoder_states_next = states[t, :, :, :, -1].reshape(self.emulator_counts, 84, 84, 1)
            actions_t = actions[t, :, :]

            all_states = np.concatenate((autoencoder_states_prev, autoencoder_states_next))
            feed_dict = {self.network.autoencoder_input_ph: all_states}
            all_latent = self.session.run(self.network.encoder_output, feed_dict=feed_dict)

            all_latent = np.split(all_latent, 2)
            prev_latent_vars = all_latent[0]
            next_latent_vars = all_latent[1]

            feed_dict = {self.network.autoencoder_input_ph: autoencoder_states_curr,
                         self.network.autoencoder_input_next_ph: autoencoder_states_next}
            curr_latent_vars, ae_loss = self.session.run(
                [self.network.encoder_output, self.network.emulator_reconstruction_loss], feed_dict=feed_dict)

            # Calculate AE loss bonus adjustment
            ae_loss_adjustment = np.clip(np.divide(ae_loss, self.autoencoder_loss_mean), .5, 2.)

            # Calculate dynamics error
            feed_dict = {self.network.dynamics_input_prev: prev_latent_vars,
                         self.network.dynamics_input: curr_latent_vars,
                         self.network.dynamics_latent_target: next_latent_vars,
                         self.network.action_input: actions_t,
                         self.network.keep_prob: 1.}
            dynamics_loss = self.session.run(self.network.dynamics_loss, feed_dict=feed_dict)
            dynamics_loss = np.divide(dynamics_loss, self.dynamics_loss_max)

            intrinsic_reward[t] = np.multiply(dynamics_loss, ae_loss_adjustment)

        intrinsic_reward = np.clip(intrinsic_reward * self.__get_exploration_const(), -self.min_max_value,
                                   self.min_max_value)
        return intrinsic_reward

    def update_reward_bonus_surprise(self, states, actions, intrinsic_reward):
        """
        Updates the intrinsic reward bonus with respect to the uncertainty extracted from the dynamics model.
        The uncertainty is extracted by MC dropout.
        :param states: observed states
        :param actions: selected actions
        :param intrinsic_reward: the intrinsic rewards to update
        :return: the updated intrinsic reward bonuses
        """
        T = self.network.T  # number of stochastic forward passes

        for t in range(self.max_local_steps):
            autoencoder_states_prev = states[t, :, :, :, -3].reshape(self.emulator_counts, 84, 84, 1)
            autoencoder_states_curr = states[t, :, :, :, -2].reshape(self.emulator_counts, 84, 84, 1)
            autoencoder_states_next = states[t, :, :, :, -1].reshape(self.emulator_counts, 84, 84, 1)
            actions_t = actions[t, :, :]

            # This is faster if GPU has enough mem
            all_states = np.concatenate((autoencoder_states_prev, autoencoder_states_next))
            feed_dict = {self.network.autoencoder_input_ph: all_states}
            all_latent = self.session.run(self.network.encoder_output, feed_dict=feed_dict)

            all_latent = np.split(all_latent, 2)
            prev_latent_vars = all_latent[0]

            feed_dict = {self.network.autoencoder_input_ph: autoencoder_states_curr,
                         self.network.autoencoder_input_next_ph: autoencoder_states_next}
            curr_latent_vars, ae_loss = self.session.run(
                [self.network.encoder_output, self.network.emulator_reconstruction_loss], feed_dict=feed_dict)

            # Calculate AE loss bonus adjustment
            ae_loss_adjustment = np.clip(np.divide(ae_loss, self.autoencoder_loss_mean), .5, 2.)

            # Calculate MC dropout surprise
            prev_latent_repeat = np.repeat(prev_latent_vars, T, axis=0)
            curr_latent_repeat = np.repeat(curr_latent_vars, T, axis=0)
            action_repeat = np.repeat(actions_t, T, axis=0)
            feed_dict = {self.network.keep_prob: .9, self.network.dynamics_input_prev: prev_latent_repeat,
                         self.network.dynamics_input: curr_latent_repeat, self.network.action_input: action_repeat}
            variance_output = self.session.run(self.network.latent_prediction, feed_dict=feed_dict)
            variance_output = variance_output.reshape(self.emulator_counts, T, self.network.latent_shape)
            action_uncertainties = np.mean(np.multiply(np.std(variance_output, axis=1), 2), axis=1)

            # Add all intrinsic rewards and discount based on time step
            intrinsic_reward[t] = np.multiply(action_uncertainties, ae_loss_adjustment)
        intrinsic_reward = np.clip(intrinsic_reward * self.__get_exploration_const(), -self.min_max_value,
                                   self.min_max_value)
        return intrinsic_reward

    def update_reward_bonus_surprise_boot_and_mc(self, states, actions, intrinsic_reward):
        """
        Updates the intrinsic reward bonus with respect to the uncertainty extracted from the dynamics model.
        However, for this bonus type, the dynamics model is bootstrapped (contains multiple output layers).
        The uncertainty is still extracted by MC dropout.
        :param states: observed states
        :param actions: selected actions
        :param intrinsic_reward: the intrinsic rewards to update
        :return: the updated intrinsic reward bonuses
        """
        T = self.network.T  # number of stochastic forward passes
        heads = self.network.num_heads  # number of output heads

        for t in range(self.max_local_steps):
            autoencoder_states_prev = states[t, :, :, :, -3].reshape(self.emulator_counts, 84, 84, 1)
            autoencoder_states_curr = states[t, :, :, :, -2].reshape(self.emulator_counts, 84, 84, 1)
            autoencoder_states_next = states[t, :, :, :, -1].reshape(self.emulator_counts, 84, 84, 1)
            actions_t = actions[t, :, :]

            all_states = np.concatenate((autoencoder_states_prev, autoencoder_states_next))
            feed_dict = {self.network.autoencoder_input_ph: all_states}
            all_latent = self.session.run(self.network.encoder_output, feed_dict=feed_dict)

            all_latent = np.split(all_latent, 2)
            prev_latent_vars = all_latent[0]

            feed_dict = {self.network.autoencoder_input_ph: autoencoder_states_curr,
                         self.network.autoencoder_input_next_ph: autoencoder_states_next}
            curr_latent_vars, ae_loss = self.session.run(
                [self.network.encoder_output, self.network.emulator_reconstruction_loss], feed_dict=feed_dict)

            # Calculate AE loss bonus adjustment
            ae_loss_adjustment = np.clip(np.divide(ae_loss, self.autoencoder_loss_mean), .5, 2.)

            # Calculate MC dropout surprise for each head
            prev_latent_repeat = np.repeat(prev_latent_vars, T, axis=0)
            curr_latent_repeat = np.repeat(curr_latent_vars, T, axis=0)
            action_repeat = np.repeat(actions_t, T, axis=0)
            feed_dict = {self.network.keep_prob: .9, self.network.dynamics_input_prev: prev_latent_repeat,
                         self.network.dynamics_input: curr_latent_repeat, self.network.action_input: action_repeat}
            variance_output = self.session.run(self.network.latent_prediction, feed_dict=feed_dict)
            sum_action_uncertainties = np.zeros(self.emulator_counts)
            for h in range(heads):
                head_output = variance_output[h]
                head_output = head_output.reshape(self.emulator_counts, T, self.network.latent_shape)
                action_uncertainties = np.mean(np.multiply(np.std(head_output, axis=1), 2), axis=1)
                sum_action_uncertainties += action_uncertainties

            # Calculate the mean uncertainty from all heads
            action_uncertainties = np.multiply(sum_action_uncertainties, 1 / heads)

            # Add all intrinsic rewards and discount based on time step
            intrinsic_reward[t] = np.multiply(action_uncertainties, ae_loss_adjustment)

        intrinsic_reward = np.clip(intrinsic_reward * self.__get_exploration_const(), -self.min_max_value,
                                   self.min_max_value)
        return intrinsic_reward

    def train_dynamics_model(self, num_epochs=100):
        """
        Dynamically trains the deep dynamics model
        """
        autoencoder_loss_arr = []
        dynamics_loss_arr = []
        for i in range(num_epochs):
            sample = random.sample(self.replay_memory_dynamics_model, 1)[0]
            idx_batch = np.random.randint(len(sample[0]), size=32)  # Find random batch
            minibatch_prev_frames = sample[0][idx_batch]
            minibatch_curr_frames = sample[1][idx_batch]
            minibatch_next_frames = sample[2][idx_batch]
            minibatch_actions = sample[3][idx_batch]
            feed_dict = {
                self.network.autoencoder_input_ph: minibatch_curr_frames,
                self.network.autoencoder_input_next_ph: minibatch_next_frames}

            if self.global_step < self.static_ae or self.static_ae == 0:
                # Train AE
                _, autoencoder_loss, latent_vars = self.session.run(
                    [self.network.autoencoder_optimizer, self.network.autoencoder_loss,
                     self.network.encoder_output], feed_dict=feed_dict)
            else:
                # Static AE
                autoencoder_loss, latent_vars = self.session.run(
                    [self.network.autoencoder_loss, self.network.encoder_output], feed_dict=feed_dict)

            feed_dict = {
                self.network.autoencoder_input_ph: np.concatenate((minibatch_prev_frames, minibatch_next_frames))}
            prev_next_latent_vars = self.session.run(self.network.encoder_output, feed_dict=feed_dict)
            prev_next_latent_vars = np.split(prev_next_latent_vars, 2)
            prev_latent_vars = prev_next_latent_vars[0]
            next_latent_vars = prev_next_latent_vars[1]

            # Train the transition prediction model
            feed_dict = {self.network.dynamics_input_prev: prev_latent_vars,
                         self.network.dynamics_input: latent_vars,
                         self.network.dynamics_latent_target: next_latent_vars,
                         self.network.action_input: minibatch_actions,
                         self.network.keep_prob: .9}
            if self.network.num_heads == 1:
                _, dynamics_loss = self.session.run([self.network.dynamics_optimizer, self.network.dynamics_loss],
                                                    feed_dict=feed_dict)
            else:
                random_head = random.randint(0, self.network.num_heads - 1)
                _, dynamics_loss = self.session.run(
                    [self.network.dynamics_optimizer[random_head], self.network.dynamics_loss[random_head]],
                    feed_dict=feed_dict)
            autoencoder_loss_arr.append(autoencoder_loss)
            dynamics_loss_arr.append(dynamics_loss)

        return autoencoder_loss_arr, dynamics_loss_arr

    def plot_dynamics_model(self, save_imgs=True):
        """
        Generates plots for the dynamics model.
        """
        n_imgs = 4
        sample = random.sample(self.replay_memory_dynamics_model, 1)[0]
        idx_batch = np.arange(n_imgs)
        minibatch_prev_frames = sample[0][idx_batch]
        minibatch_curr_frames = sample[1][idx_batch]
        minibatch_next_frames = sample[2][idx_batch]
        minibatch_actions = sample[3][idx_batch]

        feed_dict = {self.network.autoencoder_input_ph: minibatch_curr_frames}
        autoencoder_predict, latent_vars = self.session.run(
            [self.network.autoencoder_output, self.network.encoder_output], feed_dict=feed_dict)
        plot_reconstruction_examples([minibatch_curr_frames, autoencoder_predict], nb_examples=n_imgs,
                                     show_plot=not save_imgs,
                                     save_fig=save_imgs,
                                     save_path=self.debugging_folder + '/autoencoder_imgs/' + str(self.global_step))

        prev_latent_vars = self.session.run(self.network.encoder_output,
                                            feed_dict={self.network.autoencoder_input_ph: minibatch_prev_frames})

        feed_dict = {self.network.dynamics_input_prev: prev_latent_vars, self.network.dynamics_input: latent_vars,
                     self.network.action_input: minibatch_actions, self.network.keep_prob: 1.}
        head = random.randint(0, self.network.num_heads - 1)
        predicted_vars = self.session.run(self.network.latent_prediction[head], feed_dict=feed_dict)
        predicted_vars = np.add(latent_vars, predicted_vars)

        feed_dict = {self.network.dynamics_input_prev: prev_latent_vars, self.network.dynamics_input: latent_vars,
                     self.network.action_input: minibatch_actions, self.network.keep_prob: .9}
        predicted_vars_dropout = self.session.run(self.network.latent_prediction[head], feed_dict=feed_dict)
        predicted_vars_dropout = np.add(latent_vars, predicted_vars_dropout)

        feed_dict = {self.network.decoder_input: predicted_vars}
        predicted_images = self.session.run(self.network.decoder_output, feed_dict=feed_dict)

        feed_dict = {self.network.decoder_input: predicted_vars_dropout}
        predicted_images_dropout = self.session.run(self.network.decoder_output, feed_dict=feed_dict)

        plot_reconstruction_examples(
            [minibatch_curr_frames, minibatch_next_frames, predicted_images, predicted_images_dropout],
            nb_examples=n_imgs, show_plot=not save_imgs, save_fig=save_imgs,
            save_path=self.debugging_folder + '/dynamics_imgs/' + str(self.global_step))

        possible_actions = np.eye(self.num_actions)
        repeated_latent_var = np.repeat(latent_vars, self.num_actions, axis=0)[:self.num_actions]
        prev_repeated_latent_var = np.repeat(prev_latent_vars, self.num_actions, axis=0)[:self.num_actions]
        feed_dict = {self.network.dynamics_input_prev: prev_repeated_latent_var,
                     self.network.dynamics_input: repeated_latent_var, self.network.action_input: possible_actions,
                     self.network.keep_prob: 1.}
        predicted_vars = self.session.run(self.network.latent_prediction[head], feed_dict=feed_dict)
        predicted_vars = np.add(repeated_latent_var, predicted_vars)

        feed_dict = {self.network.decoder_input: predicted_vars}
        predicted_images = self.session.run(self.network.decoder_output, feed_dict=feed_dict)
        org_img_repeat = np.repeat(minibatch_curr_frames[0], self.num_actions, axis=2).transpose(2, 0, 1)
        plot_reconstruction_examples([org_img_repeat, predicted_images], nb_examples=self.num_actions,
                                     show_plot=not save_imgs, save_fig=save_imgs,
                                     save_path=self.debugging_folder + '/dynamics_imgs/' + str(
                                      self.global_step) + '_action_variance')


def plot_reconstruction_examples(img_array, nb_examples=10, img_width=84, img_height=84, show_plot=True, save_fig=False,
                                 save_path=''):
    """
    Function for generating plots comparing images.
    """
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
