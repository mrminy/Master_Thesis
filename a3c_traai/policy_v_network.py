# A3C -- in progress!
import tensorflow as tf
from network import *


class PolicyVNetwork(Network):
    def __init__(self, conf):
        """ Set up remaining layers, objective and loss functions, gradient 
        compute and apply ops, network parameter synchronization ops, and 
        summary ops. """

        super(PolicyVNetwork, self).__init__(conf)

        self.entropy_regularisation_strength = \
            conf['args'].entropy_regularisation_strength

        # Toggle additional recurrent layer
        recurrent_layer = False

        with tf.name_scope(self.name):

            self.critic_target_ph = tf.placeholder(
                "float32", [None], name='target')
            self.adv_actor_ph = tf.placeholder("float", [None], name='advantage')

            # LSTM layer with 256 cells
            # f = sigmoid(Wf * [h-, x] + bf) 
            # i = sigmoid(Wi * [h-, x] + bi) 
            # C' = sigmoid(Wc * [h-, x] + bc) 
            # o = sigmoid(Wo * [h-, x] + bo)
            # C = f * C_ +  i x C'
            # h = o * tan C
            # state = C
            # o4 = x
            if recurrent_layer:
                layer_name = 'lstm_layer';
                hiddens = 256;
                dim = 256
                with tf.variable_scope(self.name + '/' + layer_name) as vs:
                    self.lstm_cell = tf.nn.rnn_cell.LSTMCell(hiddens, dim)
                    self.lstm_cell_state = tf.Variable(
                        tf.zeros([1, self.lstm_cell.state_size]))
                    self.ox, self.lstm_cell_state = self.lstm_cell(
                        self.o3, self.lstm_cell_state)
                    # Get all LSTM trainable params
                    self.lstm_trainable_variables = [v for v in
                                                     tf.trainable_variables() if v.name.startswith(vs.name)]
            else:
                if self.arch == "NIPS":
                    self.ox = self.o3
                else:  # NATURE
                    self.ox = self.o4

            # Final actor layer
            layer_name = 'softmax_policy4'
            self.wpi, self.bpi, self.output_layer_pi = self._softmax(
                layer_name, self.ox, self.num_actions)

            # Avoiding log(0) by adding a very small quantity (1e-30) to output.
            self.log_output_layer_pi = tf.log(tf.add(self.output_layer_pi,
                                                     tf.constant(1e-30)), name=layer_name + '_log_policy')

            # Entropy: sum_a (-p_a ln p_a)
            self.output_layer_entropy = tf.reduce_sum(tf.multiply(tf.constant(-1.0),
                tf.multiply(self.output_layer_pi, self.log_output_layer_pi)), reduction_indices=1)

            # Final critic layer
            self.wv, self.bv, self.output_layer_v = self._fc(
                'fc_value4', self.ox, 1, activation="linear")

            if self.arch == "NIPS":
                self.params = [self.w1, self.b1, self.w2, self.b2, self.w3,
                               self.b3, self.wpi, self.bpi, self.wv, self.bv]
            else:  # NATURE
                self.params = [self.w1, self.b1, self.w2, self.b2, self.w3,
                               self.b3, self.w4, self.b4, self.wpi, self.bpi, self.wv, self.bv]

            if recurrent_layer:
                self.params += self.lstm_trainable_variables

            # Advantage critic
            self.adv_critic = tf.subtract(self.critic_target_ph, tf.reshape(self.output_layer_v, [-1]))

            # Actor objective
            # Multiply the output of the network by a one hot vector, 1 for the 
            # executed action. This will make the non-regularised objective 
            # term for non-selected actions to be zero.
            log_output_selected_action = tf.reduce_sum(
                tf.multiply(self.log_output_layer_pi, self.selected_action_ph),
                reduction_indices=1)
            actor_objective_advantage_term = tf.multiply(
                log_output_selected_action, self.adv_actor_ph)
            actor_objective_entropy_term = tf.multiply(
                self.entropy_regularisation_strength, self.output_layer_entropy)
            self.actor_objective = tf.reduce_sum(tf.multiply(
                tf.constant(-1.0), tf.add(actor_objective_advantage_term,
                                          actor_objective_entropy_term)))

            # Critic loss
            if self.clip_loss_delta > 0:
                quadratic_part = tf.minimum(tf.abs(self.adv_critic),
                                            tf.constant(self.clip_loss_delta))
                linear_part = tf.subtract(tf.abs(self.adv_critic), quadratic_part)
                # OBS! For the standard L2 loss, we should multiply by 0.5. However, the authors of the paper
                # recommend multiplying the gradients of the V function by 0.5. Thus the 0.5 
                self.critic_loss = tf.multiply(tf.constant(0.5), tf.nn.l2_loss(quadratic_part) + \
                                               self.clip_loss_delta * linear_part)
            else:
                # OBS! For the standard L2 loss, we should multiply by 0.5. However, the authors of the paper
                # recommend multiplying the gradients of the V function by 0.5. Thus the 0.5 
                self.critic_loss = tf.multiply(tf.constant(0.5), tf.nn.l2_loss(self.adv_critic))

            self.loss = self.actor_objective + self.critic_loss

            # Optimizer
            grads = tf.gradients(self.loss, self.params)

            # This is not really an operation, but a list of gradient Tensors. 
            # When calling run() on it, the value of those Tensors 
            # (i.e., of the gradients) will be calculated
            if self.clip_norm_type == 'ignore':
                # Unclipped gradients
                self.get_gradients = grads
            elif self.clip_norm_type == 'global':
                # Clip network grads by network norm
                self.get_gradients = tf.clip_by_global_norm(
                    grads, self.clip_norm)[0]
            elif self.clip_norm_type == 'local':
                # Clip layer grads by layer norm
                self.get_gradients = [tf.clip_by_norm(
                    g, self.clip_norm) for g in grads]

            # Placeholders for shared memory vars
            self.params_ph = []
            for p in self.params:
                self.params_ph.append(tf.placeholder(tf.float32,
                                                     shape=p.get_shape(),
                                                     name="shared_memory_for_{}".format(
                                                         (p.name.split("/", 1)[1]).replace(":", "_"))))

            # Ops to sync net with shared memory vars
            self.sync_with_shared_memory = []
            for i in range(0, len(self.params)):
                self.sync_with_shared_memory.append(
                    self.params[i].assign(self.params_ph[i]))
