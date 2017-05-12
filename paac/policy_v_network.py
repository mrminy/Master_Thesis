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
                self.dynamics_loss = []
                self.dynamics_optimizer = []
                for a in range(self.num_heads):
                    d_loss = tf.reduce_mean(tf.pow(tf.subtract(latent_diff, self.latent_prediction), 2))
                    self.dynamics_loss.append(d_loss)
                    self.dynamics_optimizer.append(tf.train.AdamOptimizer(0.0005).minimize(d_loss))

                if self.num_heads == 1:
                    self.dynamics_loss = self.dynamics_loss[0]
                    self.dynamics_optimizer = self.dynamics_optimizer[0]

                # Autoencoder optimizer
                self.autoencoder_optimizer = tf.train.AdamOptimizer(0.0005).minimize(self.autoencoder_loss)

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


class SurpriseExplorationNetwork(PolicyVDNetwork, DynamicsNetwork):
    pass


class NIPSPolicyVNetwork(PolicyVNetwork, NIPSNetwork):
    pass


class NaturePolicyVNetwork(PolicyVNetwork, NatureNetwork):
    pass
