import logging
import pickle
import random
import sys
import time
from scipy import ndimage, resize

from matplotlib import pyplot as plt
import matplotlib.animation as animation
import cv2
import gym
import numpy as np
from scipy.misc import imresize
from skimage.color import rgb2gray

import autoencoder_tests.keras_autoencoders as keras_autoencoders


def preprocess(observation, crop_top=True, grey_scale=True, flatten=False):
    """
    Pre-process an observation (down scale to (84,84) and make grey scale)
    """
    # Nr 1: Slow, but ok results
    # if grey_scale:
    #     observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
    #     if crop_top:
    #         observation = observation[:84, :]
    #     else:
    #         observation = observation[26:110, :]
    #     observation = np.reshape(observation, (84, 84, 1))
    # else:
    #     observation = cv2.resize(observation, (84, 110))
    #     if crop_top:
    #         observation = observation[:84, :]
    #     else:
    #         observation = observation[26:110, :]
    #     observation = np.reshape(observation, (84, 84, 3))
    # observation = np.divide(observation, 255.0)
    # if flatten:
    #     observation = observation.flatten()
    # return observation

    # Nr 2: Works very bad on MsPacMan
    # observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
    # observation = observation[26:110, :]
    # ret, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
    # return np.reshape(observation, (84, 84, 1))

    # Nr 3: Works ok for most environments and is pretty fast
    screen = np.dot(observation, np.array([.299, .587, .114])).astype(np.uint8)
    screen = ndimage.zoom(screen, (0.4, 0.525))
    if not flatten:
        return screen.reshape((84, 84, 1))
    return screen.flatten()


class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


if __name__ == '__main__':
    gym.undo_logger_setup()
    logger = logging.getLogger()
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.setLevel(logging.INFO)

    env = gym.make('MontezumaRevenge-v0')
    # env = gym.make('Pong-v0')
    # env = gym.make('MsPacman-v0')
    crop_top = True

    outdir = '/tmp/random-agent-results'
    # env.monitor.start(outdir, force=True)
    env.seed(0)
    agent = RandomAgent(env.action_space)

    nb_episodes = 40
    frame_skip = 1
    reward = 0
    done = False
    render_at_all = False
    max_loss = 1.

    # Experience replay parameters
    er = []
    action_history = []
    max_er = 10000

    # Special experience replay (gathered manually)
    room1_ob_pre = None
    my_er = pickle.load(open("my_er.pickle", "rb"))
    another_room = my_er[-1]

    z_shape = 512
    autoencoder, encoder, decoder = keras_autoencoders.build_conv_combo_autoencoder()
    predictor = keras_autoencoders.build_deep_predictor(z_shape, env.action_space.n)

    # Parameters for updating the dynamics model (autoencoder and transition prediction model)
    batch_size = 32
    nb_batches = None
    update_every_nb_episode = 20
    nb_epochs = 20
    autoencoder_loss_threshold = 70.  # Max threshold for starting training the prediction model

    timestep_counter = 0
    reward_history = []
    transition_loss_history = []

    action_uncertainties = np.zeros(env.action_space.n)
    x_uncertainties = np.arange(env.action_space.n)

    if render_at_all:
        plt.ion()
        graph = plt.plot(x_uncertainties, action_uncertainties)[0]
        graph2 = plt.fill(x_uncertainties, action_uncertainties)[0]
        plt.ylim([0.0, 1.])

    start_time = time.time()

    for i in range(nb_episodes):
        print("Starting episode", (i + 1))
        if (i + 1) % update_every_nb_episode == 0 and render_at_all:
            env.render()
            render = True
        else:
            render = False
        ob = env.reset()

        if (i + 1) % update_every_nb_episode == 0:
            print("ER size:", len(er))

            np_er = np.array(er)

            max_loss = keras_autoencoders.train_autoencoder(autoencoder, np_er, np_er[:batch_size],
                                                            batch_size=batch_size, nb_batches=nb_batches,
                                                            nb_epoch=nb_epochs)
            if max_loss < autoencoder_loss_threshold:
                print("updates the transition prediction model")
                predictor_x, predictor_y = keras_autoencoders.encode_to_samples(encoder, np_er, action_history, z_shape,
                                                                                split_inputs=False)
                transition_loss_history.append(
                    keras_autoencoders.train_predictor(predictor, predictor_x, predictor_y, nb_epoch=nb_epochs,
                                                       batch_size=batch_size))
        while True:
            if timestep_counter % frame_skip == 0:
                action = agent.act(ob, reward, done)

            if render:
                env.render()
                action_uncertainties = np.random.random_integers(0, 5, len(action_uncertainties)) / 5.0
                graph.set_ydata(action_uncertainties)
                plt.draw()
                plt.pause(0.01)

            ob, reward, done, _ = env.step(action)

            if reward != 0.0:
                reward_history.append(reward)

            action_one_hot = np.zeros(env.action_space.n)
            action_one_hot[action] = 1
            action_history.append(action_one_hot)

            ob_pre = preprocess(ob, crop_top=crop_top, grey_scale=True, flatten=False)
            er.append(ob_pre)

            timestep_counter += 1

            if len(er) > max_er:
                del er[:len(er) - max_er]
                del action_history[:len(action_history) - max_er]

            if done:
                break

    env.close()
    print("Finished... Used", time.time() - start_time, "seconds")

    # Compare room1 to the never seen room2 in Montezuma's Revenge (real frame on top, predicted frame on bottom)
    room1_ob_pre = my_er[-400].reshape(84, 84, 1)
    x_input = np.array([er[0], room1_ob_pre])
    print(keras_autoencoders.predict_autoencoder(autoencoder, x_input, show_examples=True))

    # Show predicted next frame with the full dynamics model (real frame on top, predicted frame on bottom)
    x_input = np.array(er[100:110])
    encoded_x_input = encoder.predict(x_input)
    actions = np.array(action_history[100:110])
    print("Actions taken:", actions)
    x_input_decoded = decoder.predict(predictor.predict(np.concatenate((encoded_x_input, actions), axis=1)))
    images_org = np.array(er[101:111])
    images_org = np.concatenate((x_input, images_org))
    keras_autoencoders.plot_images(images_org, x_input_decoded, nb_examples=10)

    # pickle.dump(transition_loss_history, open('transition_loss_history.pickle', 'wb'))
    plt.plot(transition_loss_history)
    plt.show()

    print("Timesteps:", timestep_counter)
    print(reward_history)
    print(len(reward_history))
