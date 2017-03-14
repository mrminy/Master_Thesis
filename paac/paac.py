import time
from multiprocessing import Queue
from multiprocessing.sharedctypes import RawArray
from ctypes import c_uint, c_float
from actor_learner import *
import logging

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
                          'header': 'Relative time|Absolute time|Global Time Step|[Advantage]|[R]|[Value]|[Value Discrepancy]|[Dynamics_loss]'}]
        self.stats_logger = custom_logging.StatsLogger(logger_config, subfolder=args.debugging_folder)

    @staticmethod
    def choose_next_actions(network, num_actions, states, session):
        network_output_v, network_output_pi, network_latent_var = session.run(
            [network.output_layer_v, network.output_layer_pi, network.output],
            feed_dict={network.input_ph: states})

        # TODO change network_output_pi based on uncertainty from dynamics model

        action_indices = PAACLearner.__sample_policy_action(network_output_pi)

        new_actions = np.eye(num_actions)[action_indices]

        return new_actions, network_output_v, network_output_pi, network_latent_var

    def __choose_next_actions(self, states):
        return PAACLearner.choose_next_actions(self.network, self.num_actions, states, self.session)

    @staticmethod
    def __sample_policy_action(probs):
        """
        Sample an action from an action probability distribution output by
        the policy network.
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
        latent_vars = np.zeros((self.max_local_steps, self.emulator_counts, self.network.latent_size))
        actions = np.zeros((self.max_local_steps, self.emulator_counts, self.num_actions))
        values = np.zeros((self.max_local_steps, self.emulator_counts))
        episodes_over_masks = np.zeros((self.max_local_steps, self.emulator_counts))

        start_time = time.time()

        while self.global_step < self.max_global_steps:

            loop_start_time = time.time()

            max_local_steps = self.max_local_steps
            for t in range(max_local_steps):
                next_actions, readouts_v_t, readouts_pi_t, latent_var = self.__choose_next_actions(shared_states)
                actions_sum += next_actions
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
            flat_y_batch = y_batch.reshape(-1)
            flat_adv_batch = adv_batch.reshape(-1)
            flat_actions = actions.reshape(max_local_steps * self.emulator_counts, self.num_actions)
            flat_latent_vars = latent_vars.reshape(max_local_steps * self.emulator_counts, self.network.latent_size)
            flat_concatenated_latent_actions = np.concatenate((flat_latent_vars, flat_actions), axis=1)

            lr = self.get_lr()
            feed_dict = {self.network.input_ph: flat_states,
                         self.network.critic_target_ph: flat_y_batch,
                         self.network.selected_action_ph: flat_actions,
                         self.network.adv_actor_ph: flat_adv_batch,
                         self.learning_rate: lr}
            _, value_discrepancy = self.session.run([self.train_step, self.network.value_discrepancy],
                                                    feed_dict=feed_dict)

            # TODO wait some timesteps before starting training the prediction network?
            feed_dict = {self.network.dynamics_input: flat_concatenated_latent_actions[:-1],
                         self.network.dynamics_latent_target: flat_latent_vars[1:], self.network.keep_prob: 1.}
            _, dynamics_loss = self.session.run([self.network.dynamics_optimizer, self.network.dynamics_loss],
                                                feed_dict=feed_dict)

            counter += 1
            self.stats_logger.log('learning', self.global_step,
                                  self.stats_logger.get_stats_for_array(flat_adv_batch),
                                  self.stats_logger.get_stats_for_array(flat_y_batch), Rs_stats,
                                  self.stats_logger.get_stats_for_array(value_discrepancy), dynamics_loss)
            if counter % (2048 / self.emulator_counts) == 0:
                curr_time = time.time()
                global_steps = self.global_step
                last_ten = 0.0 if len(total_rewards) < 1 else np.mean(total_rewards[-10:])
                logging.info("Ran {} steps, at {} steps/s ({} steps/s avg), last 10 rewards avg {}"
                             .format(global_steps,
                                     self.max_local_steps * self.emulator_counts / (curr_time - loop_start_time),
                                     (global_steps - global_step_start) / (curr_time - start_time),
                                     last_ten))
            self.save_vars()

        for queue in queues:
            queue.put(None)
        self.save_vars(True)
