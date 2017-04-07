from networks import *
import tensorflow as tf


class PolicyVNetwork(Network):
    def __init__(self, conf):
        """ Set up remaining layers, objective and loss functions, gradient
        compute and apply ops, network parameter synchronization ops, and
        summary ops. """

        super(PolicyVNetwork, self).__init__(conf)

        self.entropy_regularisation_strength = conf['entropy_regularisation_strength']

        # Toggle additional recurrent layer
        with tf.device(conf['device']):
            with tf.name_scope(self.name):
                self.critic_target_ph = tf.placeholder("float32", [None], name='target')
                self.adv_actor_ph = tf.placeholder("float", [None], name='advantage')

                # Final actor layer
                layer_name = 'actor_output'
                _, _, self.output_layer_pi = softmax(layer_name, self.output, self.num_actions)
                # Final critic layer
                _, _, self.output_layer_v = fc('fc_value4', self.output, 1, activation="linear")

                # Avoiding log(0) by adding a very small quantity (1e-30) to output.
                self.log_output_layer_pi = tf.log(tf.add(self.output_layer_pi, tf.constant(1e-30)),
                                                  name=layer_name + '_log_policy')

                # Entropy: sum_a (-p_a ln p_a)
                self.output_layer_entropy = tf.reduce_sum(tf.multiply(
                    tf.constant(-1.0),
                    tf.multiply(self.output_layer_pi, self.log_output_layer_pi)), reduction_indices=1)

                self.output_layer_v = tf.reshape(self.output_layer_v, [-1])

                # Advantage critic
                self.adv_critic = tf.subtract(self.critic_target_ph, self.output_layer_v)

                # Actor objective
                # Multiply the output of the network by a one hot vector, 1 for the
                # executed action. This will make the non-regularised objective
                # term for non-selected actions to be zero.
                log_output_selected_action = tf.reduce_sum(
                    tf.multiply(self.log_output_layer_pi, self.selected_action_ph),
                    reduction_indices=1)
                self.actor_objective_advantage_term = tf.multiply(
                    log_output_selected_action, self.adv_actor_ph)
                self.actor_objective_entropy_term = tf.multiply(
                    self.entropy_regularisation_strength, self.output_layer_entropy)
                self.actor_objective = tf.reduce_sum(tf.multiply(
                    tf.constant(-1.0), tf.add(self.actor_objective_advantage_term,
                                              self.actor_objective_entropy_term)))

                self.critic_loss = tf.scalar_mul(0.5, tf.nn.l2_loss(self.adv_critic))

                self.loss = self.actor_objective + self.critic_loss
                self.loss = tf.multiply(self.loss, self.one_over_emulators)

                self.value_discrepancy = tf.add(tf.subtract(self.adv_actor_ph, self.critic_target_ph),
                                                self.output_layer_v)

                self.dynamics_loss = None
                self.dynamics_optimizer = None


class PolicyVDNetwork(Network):
    def __init__(self, conf):
        """ Set up remaining layers, objective and loss functions, gradient
        compute and apply ops, network parameter synchronization ops, and
        summary ops. """

        super(PolicyVDNetwork, self).__init__(conf)

        self.entropy_regularisation_strength = conf['entropy_regularisation_strength']

        # Toggle additional recurrent layer
        with tf.device(conf['device']):
            with tf.name_scope(self.name):
                self.critic_target_ph = tf.placeholder("float32", [None], name='target')
                self.adv_actor_ph = tf.placeholder("float", [None], name='advantage')
                self.dynamics_latent_target = tf.placeholder("float32", [None, self.latent_shape],
                                                             name='dynamics_latent_target')

                # Final actor layer
                layer_name = 'actor_output'
                _, _, self.output_layer_pi = softmax(layer_name, self.output, self.num_actions)
                # Final critic layer
                _, _, self.output_layer_v = fc('fc_value4', self.output, 1, activation="linear")

                # Avoiding log(0) by adding a very small quantity (1e-30) to output.
                self.log_output_layer_pi = tf.log(tf.add(self.output_layer_pi, tf.constant(1e-30)),
                                                  name=layer_name + '_log_policy')

                # Dynamics model ops
                latent_diff = tf.subtract(self.dynamics_latent_target, self.dynamics_input)
                self.dynamics_loss_full = tf.pow(tf.subtract(latent_diff, self.latent_prediction), 2)
                self.dynamics_loss = tf.reduce_mean(self.dynamics_loss_full)
                self.dynamics_optimizer = tf.train.AdamOptimizer().minimize(self.dynamics_loss)

                # Autoencoder model ops
                self.autoencoder_movement_focus_input_ph = tf.placeholder(tf.uint8, [None, 84, 84, 1],
                                                                          name='focus_autoencoder_input')
                self.autoencoder_movement_focus_input = tf.pow(
                    tf.add(tf.scalar_mul(1.0 / 255.0, tf.cast(self.autoencoder_movement_focus_input_ph, tf.float32)),
                           1.0), 5.0)

                # MSE autoencoder loss
                self.autoencoder_loss_full = tf.pow(
                    tf.multiply(tf.subtract(self.autoencoder_input, self.autoencoder_output),
                                self.autoencoder_movement_focus_input), 2)
                self.autoencoder_loss = tf.reduce_mean(self.autoencoder_loss_full)

                def vae_loss(x, x_decoded_mean):
                    xent_loss = 7056 * metrics.binary_crossentropy(x, x_decoded_mean)
                    kl_loss = - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
                    return xent_loss + kl_loss

                # self.autoencoder_loss = vae_loss(flatten(self.autoencoder_input), self.autoencoder_output)
                self.autoencoder_optimizer = tf.train.AdamOptimizer().minimize(self.autoencoder_loss)

                # Entropy: sum_a (-p_a ln p_a)
                self.output_layer_entropy = tf.reduce_sum(
                    tf.multiply(tf.constant(-1.0), tf.multiply(self.output_layer_pi, self.log_output_layer_pi)),
                    reduction_indices=1)

                self.output_layer_v = tf.reshape(self.output_layer_v, [-1])

                # Advantage critic
                self.adv_critic = tf.subtract(self.critic_target_ph, self.output_layer_v)

                # Actor objective
                # Multiply the output of the network by a one hot vector, 1 for the
                # executed action. This will make the non-regularised objective
                # term for non-selected actions to be zero.
                log_output_selected_action = tf.reduce_sum(
                    tf.multiply(self.log_output_layer_pi, self.selected_action_ph), reduction_indices=1)
                self.actor_objective_advantage_term = tf.multiply(log_output_selected_action, self.adv_actor_ph)
                self.actor_objective_entropy_term = tf.multiply(self.entropy_regularisation_strength,
                                                                self.output_layer_entropy)
                self.actor_objective = tf.reduce_sum(tf.multiply(tf.constant(-1.0),
                                                                 tf.add(self.actor_objective_advantage_term,
                                                                        self.actor_objective_entropy_term)))

                self.critic_loss = tf.scalar_mul(0.5, tf.nn.l2_loss(self.adv_critic))

                self.loss = self.actor_objective + self.critic_loss
                self.loss = tf.multiply(self.loss, self.one_over_emulators)

                self.value_discrepancy = tf.add(tf.subtract(self.adv_actor_ph, self.critic_target_ph),
                                                self.output_layer_v)

                # Action uncertainty with tf # TODO test this!
                # self.T = conf['T']
                # transition_predictions = tf.reshape(self.latent_prediction,
                #                                     [self.emulator_counts, self.T, self.latent_shape])
                # transition_predictions_mean, transition_predictions_var = tf.nn.moments(transition_predictions,
                #                                                                         axes=[1])
                # action_uncertainties = tf.reduce_mean(tf.multiply(transition_predictions_var, 2), axis=1)
                # norm_action_uncertainties = tf.nn.softmax(action_uncertainties)
                # mean_norm_action_uncertainties = tf.reduce_mean(norm_action_uncertainties, axis=-1)
                # self.delta_action_uncertainties = tf.subtract(norm_action_uncertainties, mean_norm_action_uncertainties)


class SurpriseExplorationNetwork(PolicyVDNetwork, DynamicsNetwork):
    pass


class NIPSPolicyVNetwork(PolicyVNetwork, NIPSNetwork):
    pass


class NaturePolicyVNetwork(PolicyVNetwork, NatureNetwork):
    pass
