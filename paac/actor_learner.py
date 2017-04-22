import numpy as np
from multiprocessing import Process
import tensorflow as tf
import logging

CHECKPOINT_INTERVAL = 500000
 

class ActorLearner(Process):
    
    def __init__(self, network, environment_creator, args):
        
        super(ActorLearner, self).__init__()

        self.global_step = 0

        self.max_local_steps = args.max_local_steps
        self.num_actions = args.num_actions
        self.initial_lr = args.initial_lr
        self.lr_annealing_steps = args.lr_annealing_steps
        self.emulator_counts = args.emulator_counts
        self.device = args.device
        self.debugging_folder = args.debugging_folder
        self.last_saving_step = 0

        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=args.alpha, epsilon=args.e)

        self.emulators = np.asarray([environment_creator.create_environment(i)
                                     for i in range(self.emulator_counts)])
        self.max_global_steps = args.max_global_steps
        self.gamma = args.gamma
        self.game = args.game
        self.network = network

        # Optimizer
        grads_and_vars = self.optimizer.compute_gradients(network.loss)

        # This is not really an operation, but a list of gradient Tensors.
        # When calling run() on it, the value of those Tensors
        # (i.e., of the gradients) will be calculated
        if args.clip_norm_type == 'ignore':
            # Unclipped gradients
            gradients = grads_and_vars
        elif args.clip_norm_type == 'global':
            # Clip network grads by network norm
            gradients = tf.clip_by_global_norm(
                [g for g, v in grads_and_vars], args.clip_norm)
            gradients = list(zip(gradients[0], [v for g, v in grads_and_vars]))
            # gradients = [v for g, v in gradients if g is not None]
        elif args.clip_norm_type == 'local':
            # Clip layer grads by layer norm
            gradients = [tf.clip_by_norm(
                g, args.clip_norm) for g in grads_and_vars]
        else:
            raise Exception('Uh oh')

        self.train_step = self.optimizer.apply_gradients(gradients)

        config = tf.ConfigProto()
        if 'gpu' in self.device:
            logging.debug('Dynamic gpu mem allocation')
            config.gpu_options.allow_growth = True

        self.session = tf.Session(config=config)

    def save_vars(self, force=False):
        if force or self.global_step - self.last_saving_step >= CHECKPOINT_INTERVAL:
            self.last_saving_step = self.global_step
            self.network.saver.save(self.session, self.debugging_folder+'/checkpoints/', global_step=self.last_saving_step)
    
    def rescale_reward(self, reward):
        """ Clip immediate reward """
        if reward > 1.0:
            reward = 1.0
        elif reward < -1.0:
            reward = -1.0
        return reward

    def init_network(self):
        return self.network.init(self.debugging_folder, self.session)

    def get_lr(self):
        if self.global_step <= self.lr_annealing_steps:
            return self.initial_lr - (self.global_step * self.initial_lr / self.lr_annealing_steps)
        else:
            return 0.0

