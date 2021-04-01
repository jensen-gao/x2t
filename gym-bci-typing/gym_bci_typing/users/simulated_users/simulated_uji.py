import numpy as np
import random
import os
from utils import uji_to_image
from tensorflow import keras
import tensorflow as tf
from gym_bci_typing.users.simulated_users.simulated_user import SimulatedUser


class SimulatedUJI(SimulatedUser):
    def __init__(self, data, drift_seed, drift_std, goals, **kwargs):
        super(SimulatedUJI, self).__init__(input_dim=(28, 28, 1), n_samples=1, **kwargs)

        config = tf.ConfigProto(
            allow_soft_placement=True,
        )
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
        tf.keras.backend.set_session(session)

        self.data = data
        self.brown = None

        self.drift_state = None
        if drift_seed is not None:
            backup = np.random.get_state()
            np.random.seed(drift_seed)
            self.drift_state = np.random.get_state()
            np.random.set_state(backup)
        self.drift_std = drift_std
        self.drifts = []
        self.baseline_model = keras.models.load_model('./models/emnist_classifier/model')
        self.goals = goals
        self.goal = None
        self.gaze = False

    def setup(self, env):
        super(SimulatedUJI, self).setup(env)

        # if using language model, filters out goal sentences containing words not in language model dictionary
        if not self.env.no_lm:
            self.goals = [goal for goal in self.goals if
                          all([word in self.env.lm.lm_vocab.vocab_trie for word in goal])]

        self.goals = [' '.join(goal) for goal in self.goals]

    def get_input(self):
        next_action = self.env.actions[self.get_next_action_index()]

        # randomly selects one of the two sets of pen strokes for a character
        strokes = random.choice(self.data[next_action])

        # updates running Brownian drift noise
        self.brown += self.get_drift_noise()
        self.drifts.append(self.brown.copy())

        noise = [self.brown for _ in strokes]

        # converts pen strokes and current noise to image
        obs = uji_to_image(strokes, noise)[:, :, None]
        return obs

    def run(self, total_timesteps=None, callback=None, tb_log_name='log', mode='m', disable_learning=False):
        self.brown = 0
        result = super(SimulatedUJI, self).run(total_timesteps, callback, tb_log_name, mode, disable_learning)

        # logs drift noise over each timestep
        with open(os.path.join(self.model.tensorboard_log, 'drifts.txt'), 'w') as f:
            for drift in self.drifts:
                f.write('%s\n' % drift)

        return result

    def get_drift_noise(self):
        # gets next seeded drift noise to update running Brownian noise
        if self.drift_state is not None:
            backup = np.random.get_state()
            np.random.set_state(self.drift_state)

        drift_noise = np.random.normal(scale=self.drift_std, size=2)

        if self.drift_state is not None:
            self.drift_state = np.random.get_state()
            np.random.set_state(backup)

        return drift_noise

    def baseline(self, obs):
        pred = self.baseline_model.predict(np.array(obs))[0]
        return pred, None

    def get_next_action_index(self):
        """
        Gets the next desired action index according to the user's goal.
        """
        next_index = len(''.join(self.env.typed))

        goal_char = self.goal[next_index]
        return self.env.actions.index(goal_char)

    def reset(self):
        self.goal = random.choice(self.goals)
