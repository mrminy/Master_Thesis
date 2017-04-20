from train import get_network_and_environment_creator, bool_arg
import custom_logging
import argparse
import numpy as np
import time
from PIL import Image
import tensorflow as tf
from paac import PAACLearner


def get_save_frame(dest_folder, name):
    class Counter:
        def __init__(self):
            self.counter = 0
        def increase(self):
            self.counter += 1
        def get(self):
            return self.counter
    counter = Counter()

    def get_frame(frame):
        im = Image.fromarray(frame[:, :, ::-1])
        im.save("{}/{}_{:05d}.png".format(dest_folder, name, counter.get()))
        counter.increase()
        return False
    return get_frame

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', default='debugging/', type=str, help="Folder where to save the debugging information.", dest="folder")
    parser.add_argument('-tc', '--test_count', default='30', type=int, help="The amount of tests to run on the given network", dest="test_count")
    parser.add_argument('-re', '--random_eval', default=False, type=bool_arg, help="Whether or not to use 35 random steps", dest="random_eval")
    parser.add_argument('-s', '--show', default=False, type=bool_arg, help="Whether or not to show the run", dest="show")
    parser.add_argument('-gf', '--gif_folder', default=None, type=str, help="The folder to save the gifs", dest="gif_folder")

    args = parser.parse_args()
    arg_file = args.folder+'args.json'
    for k, v in custom_logging.load_args(arg_file).items():
        setattr(args, k, v)
    args.max_global_steps = 0
    df = args.debugging_folder
    args.debugging_folder = '/tmp/logs'
    args.device = '/gpu:0'

    if args.random_eval:
        args.random_start = False
    args.single_life_episodes = False
    if args.show:
        args.visualize = 1

    args.actor_id = 0
    rng = np.random.RandomState(int(time.time()))
    args.random_seed = rng.randint(1000)

    network, env_creator = get_network_and_environment_creator(args)

    rewards = []
    environment = env_creator.create_environment(0)
    if args.gif_folder:
        environment.on_new_frame = get_save_frame(args.gif_folder, 'gif')
    print(args.random_eval)
    with tf.Session() as sess:
        network.init(df, sess)
        for i in range(args.test_count):
            state = environment.get_initial_state()
            if args.random_eval:
                for _ in range(35):
                    state, _, _ = environment.next(np.eye(environment.get_legal_actions().shape[0])[rng.randint(environment.get_legal_actions().shape[0])])

            episode_over = False
            reward = 0.0
            while not episode_over:
                action = PAACLearner.choose_next_actions(network, env_creator.num_actions, [state], sess)
                state, r, episode_over = environment.next(action[0])
                reward += r
            rewards.append(reward)
            print(reward)
        print("Mean:", np.mean(rewards), "Min:", np.min(rewards), "Max:", np.max(rewards), "Std:", np.std(rewards))


