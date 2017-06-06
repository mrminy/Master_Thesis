"""
Specialized script for gathering data from Montezuma's Revenge
Keys:
    A - left
    D - right
    S - down
    W - up
    E - jump right
    Q - jump left
"""

import pickle
import gym
import numpy as np

from autoencoder_experiments.random_agent import preprocess

# Parameters
env = gym.make('MontezumaRevenge-v0')
ACTIONS = env.action_space.n
SKIP_CONTROL = 0
max_play_time = 10000
human_agent_action = 0
human_wants_restart = False
human_sets_pause = False
save_er = False  # Set true to save a observations to pickle


def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause
    if key == 0xff0d: human_wants_restart = True
    if key == 32: human_sets_pause = not human_sets_pause
    a = int(key - ord('0'))
    if key == 115:  # S
        a = 5  # down
    elif key == 119:  # W
        a = 2  # up
    elif key == 101:  # E
        a = 14  # right jump
    elif key == 113:  # Q
        a = 15  # left-jump
    elif key == 100:  # D
        a = 3  # right
    elif key == 97:  # A
        a = 4  # left
    if a <= 0 or a >= ACTIONS: return
    human_agent_action = a


def key_release(key, mod):
    global human_agent_action
    a = int(key - ord('0'))
    if key == 115 or key == 119 or key == 101 or key == 113 or key == 100 or key == 97:
        human_agent_action = 0
    if a <= 0 or a >= ACTIONS: return
    if human_agent_action == a:
        human_agent_action = 0


env.render()
env.unwrapped.viewer.window.on_key_press = key_press
env.unwrapped.viewer.window.on_key_release = key_release


def rollout(env):
    global human_agent_action, human_wants_restart, human_sets_pause
    human_wants_restart = False
    a = 0
    er = []
    obser = env.reset()
    skip = 4
    for t in range(max_play_time):
        if not skip:
            if human_agent_action != 0:
                a = human_agent_action
            else:
                a = 0
            skip = SKIP_CONTROL
        else:
            skip -= 1

        obser, r, done, info = env.step(a)
        obs_pre = preprocess(obser, flatten=False)

        er.append(obs_pre)
        env.render()
        if done: break
        if human_wants_restart: break
        while human_sets_pause:
            env.render()
            import time
            time.sleep(0.1)

    return er


# Play an episode and collect experience
er = rollout(env)
if save_er:
    pickle.dump(np.array(er), open("er.pickle", "wb"))
