import os
import sys
import tempfile
import unittest
from inspect import getsourcefile

import gym
import tensorflow as tf

from a3c.estimators import PolicyEstimator, ValueEstimator
from a3c.policy_monitor import PolicyMonitor

current_path = os.path.dirname(os.path.abspath(getsourcefile(lambda: 0)))
import_path = os.path.abspath(os.path.join(current_path, "../.."))

if import_path not in sys.path:
    sys.path.append(import_path)


def make_env():
    return gym.envs.make("Breakout-v0")


VALID_ACTIONS = [0, 1, 2, 3]


class PolicyMonitorTest(tf.test.TestCase):
    def setUp(self):
        super(PolicyMonitorTest, self).setUp()

        self.env = make_env()
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.summary_writer = tf.summary.FileWriter(tempfile.mkdtemp())

        with tf.variable_scope("global") as vs:
            self.global_policy_net = PolicyEstimator(len(VALID_ACTIONS))
            self.global_value_net = ValueEstimator(reuse=True)

    def testEvalOnce(self):
        pe = PolicyMonitor(env=self.env, policy_net=self.global_policy_net, summary_writer=self.summary_writer)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            total_reward, episode_length = pe.eval_once(sess)
            self.assertTrue(episode_length > 0)


if __name__ == '__main__':
    unittest.main()
