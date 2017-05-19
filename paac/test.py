import random
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import TSNE
from train import get_network_and_environment_creator, bool_arg
import custom_logging
import argparse
import numpy as np
import time
from PIL import Image
import tensorflow as tf
from paac import PAACLearner


def imscatter(x, y, image, ax=None, zoom=1.):
    if ax is None:
        ax = plt.gca()
    im = OffsetImage(image, zoom=zoom, cmap='gray')
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists


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
    parser.add_argument('-f', '--folder', default='debugging/', type=str,
                        help="Folder where to save the debugging information.", dest="folder")
    parser.add_argument('-tc', '--test_count', default='30', type=int,
                        help="The amount of tests to run on the given network", dest="test_count")
    parser.add_argument('-re', '--random_eval', default=False, type=bool_arg,
                        help="Whether or not to use 35 random steps", dest="random_eval")
    parser.add_argument('-s', '--show', default=False, type=bool_arg, help="Whether or not to show the run",
                        dest="show")
    parser.add_argument('-ep', '--embedding_plot', default=False, type=bool_arg,
                        help="Whether or not to show the TSNE embedding with images",
                        dest="embedding_plot")
    parser.add_argument('-gf', '--gif_folder', default=None, type=str, help="The folder to save the gifs",
                        dest="gif_folder")

    args = parser.parse_args()
    arg_file = args.folder + 'args.json'
    for k, v in custom_logging.load_args(arg_file).items():
        setattr(args, k, v)
    args.max_global_steps = 0
    df = args.folder
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

    db_size = 5000
    img_database = deque(maxlen=db_size)
    latent_database = deque(maxlen=db_size)

    network, env_creator = get_network_and_environment_creator(args)

    rewards = []
    environment = env_creator.create_environment(0)
    if args.gif_folder:
        environment.on_new_frame = get_save_frame(args.gif_folder, 'gif')

    config = config = tf.ConfigProto()
    if 'gpu' in args.device:
        config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        network.init(df, sess)

        for i in range(args.test_count):
            state = environment.get_initial_state()
            if args.random_eval:
                for _ in range(35):
                    state, _, _ = environment.next(np.eye(environment.get_legal_actions().shape[0])[
                                                       rng.randint(environment.get_legal_actions().shape[0])])

            episode_over = False
            reward = 0.0
            while not episode_over:
                if random.random() < 0.2:
                    latent_var = sess.run(network.encoder_output, feed_dict={
                        network.autoencoder_input_ph: np.array([state[:, :, 0]]).reshape(1, 84, 84, 1)})
                    img_database.append(state[:, :, 0])
                    latent_database.append(latent_var[0])

                action = PAACLearner.choose_next_actions(network, env_creator.num_actions, [state], sess)
                state, r, episode_over = environment.next(action[0])
                reward += r
            rewards.append(reward)
            print(reward)
        print("Mean:", np.mean(rewards), "Min:", np.min(rewards), "Max:", np.max(rewards), "Std:", np.std(rewards))

    if args.embedding_plot:
        manifold = TSNE(n_components=2)
        x_fitted = manifold.fit(np.array(latent_database))
        print("Plotting")
        fig, ax = plt.subplots()
        num_imgs = 30
        idx = np.random.randint(len(img_database), size=num_imgs)  # Find random batch

        for i in range(num_imgs):
            x = x_fitted.embedding_[idx[i]][0]
            y = x_fitted.embedding_[idx[i]][1]
            img = img_database[idx[i]]
            imscatter(x, y, img, ax=ax, zoom=.7)
            ax.plot(x, y)
        plt.show()
