import pickle

import gym
import numpy as np

from autoencoder_tests.random_agent import preprocess

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

env = gym.make('MontezumaRevenge-v0')
ACTIONS = env.action_space.n
ROLLOUT_TIME = 10000
SKIP_CONTROL = 0
human_agent_action = 0
human_wants_restart = False
human_sets_pause = False
exit_and_save = False


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
    elif key == 65307:
        exit_and_save = True
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
    for t in range(ROLLOUT_TIME):
        if not skip:
            # print("taking action {}".format(human_agent_action))
            # a = random.randint(9,17)
            if human_agent_action != 0:
                a = human_agent_action
            else:
                a = 0
            skip = SKIP_CONTROL
        else:
            skip -= 1

        obser, r, done, info = env.step(a)
        obs_pre = preprocess(obser, crop_top=True, grey_scale=True, flatten=False)

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
pickle.dump(np.array(er), open("my_er.pickle", "wb"))
