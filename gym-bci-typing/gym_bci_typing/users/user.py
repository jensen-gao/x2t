from abc import ABC, abstractmethod
import numpy as np


class User(ABC):
    """
    User in the typing environment. Can be simulated or a real user.

    :param input_dim: (Tuple(int)) The dimensionality of the inputs this users produces.
    """
    def __init__(self, input_dim, n_samples, baseline_temp, boltzmann_exploration):
        self.env = None
        self.input_dim = input_dim
        self.n_samples = n_samples
        self.model = None
        self.baseline_temp = baseline_temp
        self.boltzmann_exploration = boltzmann_exploration

    def setup(self, env):
        """
        Sets up the environment with the user. Must be called before running the user.
        """
        self.env = env

    def setup_model(self, model):
        """
        Sets up the predictive model. Should be called by the model during initialization.
        """
        self.model = model

    @abstractmethod
    def run(self, total_timesteps, callback, tb_log_name, mode, disable_learning):
        """
        Runs the user with the typing environment.
        """
        pass

    def predict(self, obs):
        """
        Returns the predictive model's distribution over actions given inputs.
        """
        with self.model.sess.as_default():
            obs = np.array(obs)[None]
            pred = self.model.act(obs, self.env.targets)
        pred = np.squeeze(pred)
        return pred

    def reset(self):
        pass

    @abstractmethod
    def get_input(self):
        """
        Gets the input at the current timestep
        """
        pass

    @abstractmethod
    def baseline(self, obs):
        """
        Baseline predictive method
        """
        pass

    @abstractmethod
    def get_next_action_index(self):
        """
        Gets the next desired action index according to the user's goal.
        """
        pass

    @staticmethod
    def softmax(x):
        unnormalized = np.exp(x)
        return unnormalized / np.sum(unnormalized)
