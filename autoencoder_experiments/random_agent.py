"""
Test script for experiments with the deep dynamics model. Contains a random agent for playing Atari games from OpenAI.
Together with the random agent, it trains a deep dynamics model dynamically while the random agents plays.
"""

import logging
import pickle
import sys
import time

import gym
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

from autoencoder_experiments.keras_autoencoders import build_deep_predictor, build_conv_combo_autoencoder, \
    train_autoencoder, encode_to_samples, train_predictor, predict_autoencoder, plot_images


def preprocess(observation, flatten=False):
    """
    Pre-process an observation (down scale to (84,84) and make grey scale)
    """
    screen = np.dot(observation, np.array([.299, .587, .114])).astype(np.uint8)
    screen = ndimage.zoom(screen, (0.4, 0.525))
    if not flatten:
        return screen.reshape((84, 84, 1))
    return screen.flatten()


class RandomAgent(object):
    """
    A random agent for Atari from OpenAI
    """

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

    env = gym.make('Pong-v0') # Select environment
    # env = gym.make('MontezumaRevenge-v0')

    outdir = '/tmp/random-agent-results'
    env.seed(0)
    agent = RandomAgent(env.action_space)

    nb_episodes = 40 # Select number of episodes to play
    frame_skip = 1
    reward = 0
    done = False
    render_at_all = False
    max_loss = 1.

    # Experience replay parameters
    max_er = 10000 # Select experience replay size
    er = []
    action_history = []

    # Special experience replay (gathered manually)
    room1_ob_pre = None
    my_er = pickle.load(open("er.pickle", "rb"))
    another_room = my_er[-1]

    z_shape = 256 # Select latent space size
    autoencoder, encoder, decoder = build_conv_combo_autoencoder()
    predictor = build_deep_predictor(z_shape, env.action_space.n)

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

    start_time = time.time()

    for i in range(nb_episodes):
        print("Starting episode", (i + 1))
        ob = env.reset()

        if (i + 1) % update_every_nb_episode == 0:
            print("ER size:", len(er))

            np_er = np.array(er)

            max_loss = train_autoencoder(autoencoder, np_er, np_er[:batch_size],
                                         batch_size=batch_size, nb_batches=nb_batches,
                                         nb_epoch=nb_epochs)
            if max_loss < autoencoder_loss_threshold:
                print("updates the transition prediction model")
                predictor_x, predictor_y = encode_to_samples(encoder, np_er, action_history, z_shape,
                                                             split_inputs=False)
                transition_loss_history.append(train_predictor(predictor, predictor_x, predictor_y, nb_epoch=nb_epochs,
                                                               batch_size=batch_size))
        while True:
            if timestep_counter % frame_skip == 0:
                action = agent.act(ob, reward, done)

            ob, reward, done, _ = env.step(action)

            if reward != 0.0:
                reward_history.append(reward)

            action_one_hot = np.zeros(env.action_space.n)
            action_one_hot[action] = 1
            action_history.append(action_one_hot)

            ob_pre = preprocess(ob, flatten=False)
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
    print(predict_autoencoder(autoencoder, x_input, show_examples=True))

    # Show predicted next frame with the full dynamics model (real frame on top, predicted frame on bottom)
    x_input = np.array(er[100:110])
    encoded_x_input = encoder.predict(x_input)
    actions = np.array(action_history[100:110])
    print("Actions taken:", actions)
    x_input_decoded = decoder.predict(predictor.predict(np.concatenate((encoded_x_input, actions), axis=1)))
    images_org = np.array(er[101:111])
    images_org = np.concatenate((x_input, images_org))
    plot_images(images_org, x_input_decoded, nb_examples=10)

    plt.plot(transition_loss_history)
    plt.show()

    print("Timesteps:", timestep_counter)
    print(reward_history)
    print(len(reward_history))
