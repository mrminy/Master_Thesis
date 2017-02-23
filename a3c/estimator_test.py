import unittest
import gym
import sys
import os
import numpy as np
import tensorflow as tf

from inspect import getsourcefile

from a3c.estimators import PolicyEstimator, ValueEstimator
from a3c.helpers import atari_make_initial_state
from a3c.state_processor import StateProcessor

current_path = os.path.dirname(os.path.abspath(getsourcefile(lambda: 0)))
import_path = os.path.abspath(os.path.join(current_path, "../.."))

if import_path not in sys.path:
    sys.path.append(import_path)


def make_env():
    return gym.envs.make("Breakout-v0")


VALID_ACTIONS = [0, 1, 2, 3]


class PolicyEstimatorTest(tf.test.TestCase):
    def testPredict(self):
        env = make_env()
        sp = StateProcessor()
        estimator = PolicyEstimator(len(VALID_ACTIONS))

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            # Generate a state
            state = sp.process(env.reset())
            processed_state = atari_make_initial_state(state)
            processed_states = np.array([processed_state])

            # Run feeds
            feed_dict = {
                estimator.states: processed_states,
                estimator.targets: [1.0],
                estimator.actions: [1]
            }
            loss = sess.run(estimator.loss, feed_dict)
            pred = sess.run(estimator.predictions, feed_dict)

            # Assertions
            self.assertTrue(loss != 0.0)
            self.assertEqual(pred["probs"].shape, (1, len(VALID_ACTIONS)))
            self.assertEqual(pred["logits"].shape, (1, len(VALID_ACTIONS)))

    def testGradient(self):
        env = make_env()
        sp = StateProcessor()
        estimator = PolicyEstimator(len(VALID_ACTIONS))
        grads = [g for g, _ in estimator.grads_and_vars]

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            # Generate a state
            state = sp.process(env.reset())
            processed_state = atari_make_initial_state(state)
            processed_states = np.array([processed_state])

            # Run feeds to get gradients
            feed_dict = {
                estimator.states: processed_states,
                estimator.targets: [1.0],
                estimator.actions: [1]
            }
            grads_ = sess.run(grads, feed_dict)

            # Apply calculated gradients
            grad_feed_dict = {k: v for k, v in zip(grads, grads_)}
            _ = sess.run(estimator.train_op, grad_feed_dict)


class ValueEstimatorTest(tf.test.TestCase):
    def testPredict(self):
        env = make_env()
        sp = StateProcessor()
        estimator = ValueEstimator()

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            # Generate a state
            state = sp.process(env.reset())
            processed_state = atari_make_initial_state(state)
            processed_states = np.array([processed_state])

            # Run feeds
            feed_dict = {
                estimator.states: processed_states,
                estimator.targets: [1.0],
            }
            loss = sess.run(estimator.loss, feed_dict)
            pred = sess.run(estimator.predictions, feed_dict)

            # Assertions
            self.assertTrue(loss != 0.0)
            self.assertEqual(pred["logits"].shape, (1,))

    def testGradient(self):
        env = make_env()
        sp = StateProcessor()
        estimator = ValueEstimator()
        grads = [g for g, _ in estimator.grads_and_vars]

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            # Generate a state
            state = sp.process(env.reset())
            processed_state = atari_make_initial_state(state)
            processed_states = np.array([processed_state])

            # Run feeds
            feed_dict = {
                estimator.states: processed_states,
                estimator.targets: [1.0],
            }
            grads_ = sess.run(grads, feed_dict)

            # Apply calculated gradients
            grad_feed_dict = {k: v for k, v in zip(grads, grads_)}
            _ = sess.run(estimator.train_op, grad_feed_dict)


if __name__ == '__main__':
    unittest.main()
