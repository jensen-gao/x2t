import tensorflow as tf
import numpy as np

from stable_baselines.deepq.policies import DQNPolicy
from stable_baselines.common.tf_layers import conv, conv_to_fc


class TypingPolicy(DQNPolicy):
    """
    Policy object for use with X2T and typing envs.

    Code adapted from Stable Baselines 2.
    """
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False,
                 units=64, mode='gaze', keep_prob=1, **_kwargs):
        super(TypingPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps,
                                           n_batch, reuse=reuse, scale=False, obs_phs=None, **_kwargs)
        self.units = units
        self._obs_ph = tf.placeholder(shape=(None, None,) + ob_space.shape, dtype=ob_space.dtype, name='obs')

        # 2D coordinates of actions in the radial typing env
        self.targets_ph = tf.placeholder(shape=(ac_space.n, 2), dtype=tf.float32, name='targets')

        self.train_ph = tf.placeholder(shape=(), dtype=tf.bool)

        with tf.variable_scope("model", reuse=reuse):
            if mode == 'gaze':
                hidden = tf.layers.dense(self._obs_ph, self.units, activation=tf.nn.relu)
                hidden = tf.cond(self.train_ph, lambda: tf.nn.dropout(hidden, keep_prob), lambda: hidden)

                # model's 2D gaze estimates
                locations = tf.layers.dense(hidden, 2)

                # average gaze estimate across all inputs
                averages = tf.math.reduce_mean(locations, axis=1)

                # q_values are negative distances from average estimate to each of the action locations
                self.q_values = -tf.norm(averages[:, None] - self.targets_ph[None], axis=-1)
            elif mode == 'char':
                conv1 = tf.nn.relu(conv(tf.squeeze(self._obs_ph, axis=1),
                                        'conv1', n_filters=32, filter_size=5, stride=1))
                p1 = tf.nn.max_pool2d(conv1, 2, 2, 'VALID')
                d1 = tf.cond(self.train_ph, lambda: tf.nn.dropout(p1, keep_prob), lambda: p1)
                conv2 = tf.nn.relu(conv(d1, 'conv2', n_filters=64, filter_size=5, stride=1))
                p2 = tf.nn.max_pool2d(conv2, 2, 2, 'VALID')
                d2 = tf.cond(self.train_ph, lambda: tf.nn.dropout(p2, keep_prob), lambda: p2)

                flattened = conv_to_fc(d2)
                self.q_values = tf.layers.dense(flattened, ac_space.n, name='q_values')

        self.softmax = tf.nn.softmax(self.q_values)
        self._setup_init()

    def step(self, obs_seq, state=None, mask=None, deterministic=True):
        q_values, actions_proba = self.sess.run([self.q_values, self.policy_proba], {self._obs_ph: obs_seq})

        actions = self.select_actions(deterministic, actions_proba)
        return actions, q_values

    def select_actions(self, deterministic, actions_proba):
        if deterministic:
            actions = np.argmax(actions_proba, axis=1)
        else:
            # Unefficient sampling
            # TODO: replace the loop
            # maybe with Gumbel-max trick ? (http://amid.fish/humble-gumbel)
            actions = np.zeros(len(actions_proba), dtype=np.int64)
            for action_idx in range(len(actions_proba)):
                actions[action_idx] = np.random.choice(self.n_actions, p=actions_proba[action_idx])
        return actions

    def proba_step(self, obs_seq, state=None, mask=None):
        return self.sess.run([self.policy_proba], {self._obs_ph: obs_seq})

    def _setup_init(self):
        """
        Set up action probability
        """
        with tf.variable_scope("output", reuse=True):
            assert self.q_values is not None
            self.policy_proba = self.softmax
