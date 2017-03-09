import numpy as np
from multiprocessing import Process 
import logging_utils
import tensorflow as tf
import ctypes
import pyximport; pyximport.install()
from hogupdatemv import copy, apply_grads_mom_rmsprop, apply_grads_adam
import time
import utils

CHECKPOINT_INTERVAL = 500000
 
logger = logging_utils.getLogger('actor_learner')

def generate_final_epsilon():
    """ Generate lower limit for decaying epsilon. """
    epsilon = {'limits': [0.1, 0.01, 0.5], 'probs': [0.4, 0.3, 0.3]}
    return np.random.choice(epsilon['limits'], p=epsilon['probs']) 

class ActorLearner(Process):
    
    def __init__(self, args):
        
        super(ActorLearner, self).__init__()
       
        self.summ_base_dir = args.summ_base_dir
        
        self.local_step = 0
        self.global_step = args.global_step

        self.actor_id = args.actor_id
        self.alg_type = args.alg_type
        self.max_local_steps = args.max_local_steps
        self.optimizer_type = args.opt_type
        self.optimizer_mode = args.opt_mode
        self.num_actions = args.num_actions
        self.initial_lr = args.initial_lr
        self.lr_annealing_steps = args.lr_annealing_steps 
        
        # Shared mem vars
        self.learning_vars = args.learning_vars
        size = self.learning_vars.size
        self.flat_grads = np.empty(size, dtype = ctypes.c_float)
            
        if (self.optimizer_mode == "local"):
                if (self.optimizer_type == "rmsprop"):
                    self.opt_st = np.ones(size, dtype = ctypes.c_float)
                else:
                    self.opt_st = np.zeros(size, dtype = ctypes.c_float)
        elif (self.optimizer_mode == "shared"):
                self.opt_st = args.opt_state

        # rmsprop/momentum
        self.alpha = args.alpha
        # adam
        self.b1 = args.b1
        self.b2 = args.b2
        self.e = args.e
        
        if args.env == "GYM":
            from atari_environment import AtariEnvironment
            self.emulator = AtariEnvironment(args.game, args.visualize)
        else:
            from emulator import Emulator
            self.emulator = Emulator(
                args.rom_path, 
                args.game, 
                args.visualize, 
                self.actor_id,
                args.random_seed,
                args.single_life_episodes)
            
        self.grads_update_steps = args.grads_update_steps
        self.max_global_steps = args.max_global_steps
        self.gamma = args.gamma

        # Exploration epsilons 
        self.epsilon = 1.0
        self.initial_epsilon = 1.0
        self.final_epsilon = generate_final_epsilon()
        self.epsilon_annealing_steps = args.epsilon_annealing_steps

        self.rescale_rewards = args.rescale_rewards
        self.max_achieved_reward = -1000000
        if self.rescale_rewards:
            self.thread_max_reward = 1.0

        # Barrier to synchronize all actors after initialization is done
        self.barrier = args.barrier
        
        self.summary_ph, self.update_ops, self.summary_ops = self.setup_summaries()
        self.game = args.game
        

    def run(self):
        # config = tf.ConfigProto(device_count={'GPU': 0})
        # self.session = tf.Session(config=config)
        self.session = tf.Session()

        if (self.actor_id==0):
            #Initizlize Tensorboard summaries
            self.summary_op = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(
                            "{}/{}".format(self.summ_base_dir, self.actor_id), self.session.graph_def) 

            # Initialize network parameters
            g_step = utils.restore_vars(self.saver, self.session, self.game, self.alg_type, self.max_local_steps)
            self.global_step.val.value = g_step
            self.last_saving_step = g_step   
            logger.debug("T{}: Initializing shared memory...".format(self.actor_id))
            self.init_shared_memory()

        # Wait until actor 0 finishes initializing shared memory
        self.barrier.wait()
        
        if self.actor_id > 0:
            logger.debug("T{}: Syncing with shared memory...".format(self.actor_id))
            self.sync_net_with_shared_memory(self.local_network, self.learning_vars)  
            if self.alg_type != "a3c":
                self.sync_net_with_shared_memory(self.target_network, self.target_vars)

        # Wait until all actors are ready to start 
        self.barrier.wait()
        
        # Introduce a different start delay for each actor, so that they do not run in synchronism.
        # This is to avoid concurrent updates of parameters as much as possible 
        time.sleep(0.1877 * self.actor_id)

    def save_vars(self):
        if (self.actor_id == 0 and 
            (self.global_step.value() - self.last_saving_step >= CHECKPOINT_INTERVAL)):
            self.last_saving_step = self.global_step.value()
            utils.save_vars(self.saver, self.session, self.game, self.alg_type, self.max_local_steps, self.last_saving_step) 
    
    def init_shared_memory(self):
        # Initialize shared memory with tensorflow var values
        params = self.session.run(self.local_network.params)            
        # Merge all param matrices into a single 1-D array
        params = np.hstack([p.reshape(-1) for p in params])
        np.frombuffer(self.learning_vars.vars, ctypes.c_float)[:] = params
        if self.alg_type != "a3c":
            np.frombuffer(self.target_vars.vars, ctypes.c_float)[:] = params
        #memoryview(self.learning_vars.vars)[:] = params
        #memoryview(self.target_vars.vars)[:] = memoryview(self.learning_vars.vars)
    
    def reduce_thread_epsilon(self):
        """ Linear annealing """
        if self.epsilon > self.final_epsilon:
                self.epsilon -= (self.initial_epsilon - self.final_epsilon) / self.epsilon_annealing_steps

    
    def apply_gradients_to_shared_memory_vars(self, grads):
            #Flatten grads
            offset = 0
            for g in grads:
                self.flat_grads[offset:offset + g.size] = g.reshape(-1)
                offset += g.size
            g = self.flat_grads
            
            if self.optimizer_type == "adam" and self.optimizer_mode == "shared":
                p = np.frombuffer(self.learning_vars.vars, ctypes.c_float)
                p_size = self.learning_vars.size
                m = np.frombuffer(self.opt_st.ms, ctypes.c_float)
                v = np.frombuffer(self.opt_st.vs, ctypes.c_float)
                T = self.global_step.value() 
                self.opt_st.lr.value =  1.0 * self.opt_st.lr.value * (1 - self.b2**T)**0.5 / (1 - self.b1**T) 
                
                apply_grads_adam(m, v, g, p, p_size, self.opt_st.lr.value, self.b1, self.b2, self.e)
                    
            else: #local or shared rmsprop/momentum
                lr = self.decay_lr()
                if (self.optimizer_mode == "local"):
                    m = self.opt_st
                else: #shared 
                    m = np.frombuffer(self.opt_st.vars, ctypes.c_float)
                
                p = np.frombuffer(self.learning_vars.vars, ctypes.c_float)
                p_size = self.learning_vars.size
                _type = 0 if self.optimizer_type == "momentum" else 1
                    
                #print "BEFORE", "RMSPROP m", m[0], "GRAD", g[0], self.flat_grads[0], self.flat_grads2[0]
                apply_grads_mom_rmsprop(m, g, p, p_size, _type, lr, self.alpha, self.e)
                #print "AFTER", "RMSPROP m", m[0], "GRAD", g[0], self.flat_grads[0], self.flat_grads2[0]

    def rescale_reward(self, reward):
        if self.rescale_rewards:
            """ Rescale immediate reward by max reward encountered thus far. """
            if reward > self.thread_max_reward:
                self.thread_max_reward = reward
            return reward/self.thread_max_reward
        else:
            """ Clip immediate reward """
            if reward > 1.0:
                reward = 1.0
            elif reward < -1.0:
                reward = -1.0
            return reward
            

    def sync_net_with_shared_memory(self, dest_net, shared_mem_vars):
        feed_dict = {}
        offset = 0
        params = np.frombuffer(shared_mem_vars.vars, 
                                  ctypes.c_float)
        for i in range(0,len(dest_net.params)):
            shape = shared_mem_vars.var_shapes[i]
            size = np.prod(shape)
            feed_dict[dest_net.params_ph[i]] = \
                    params[offset:offset+size].reshape(shape)
            offset += size
        
        self.session.run(dest_net.sync_with_shared_memory, 
                feed_dict=feed_dict)


    def decay_lr(self):
        if self.global_step.value() <= self.lr_annealing_steps:            
            return self.initial_lr - (self.global_step.value() * self.initial_lr / self.lr_annealing_steps)
        else:
            return 0.0

    
    def setup_summaries(self):
        episode_reward = tf.Variable(0.)
        s1 = tf.summary.scalar("Episode Reward " + str(self.actor_id), episode_reward)
        if self.alg_type == "a3c":
            summary_vars = [episode_reward]
        else:
            episode_ave_max_q = tf.Variable(0.)
            s2 = tf.summary.scalar("Max Q Value " + str(self.actor_id), episode_ave_max_q)
            logged_epsilon = tf.Variable(0.)
            s3 = tf.summary.scalar("Epsilon " + str(self.actor_id), logged_epsilon)
            summary_vars = [episode_reward, episode_ave_max_q, logged_epsilon]
        summary_placeholders = [tf.placeholder("float") for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        with tf.control_dependencies(update_ops):
            summary_ops = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_ops
    
