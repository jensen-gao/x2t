import numpy as np
from gym_bci_typing.users import *
from stable_baselines.common import TensorboardWriter
from abc import ABC


class SimulatedUser(User, ABC):
    """
    Simulated user for the typing environment.
    """
    def __init__(self, error_rate=0, **kwargs):
        super(SimulatedUser, self).__init__(**kwargs)
        self.sample_index = None
        self.error_rate = error_rate
        self.next_action_index = None
        self.done = False
        self.gaze = True

    def run(self, total_timesteps=None, callback=None, tb_log_name='log', mode='m', disable_learning=False):
        """
        Runs the user in the typing environment.
        """
        self.model.setup_learn(callback=callback)
        self.env.reset()

        successes = 0
        n_episodes = 0

        baseline_estimates = [[] for _ in range(self.env.action_space.n)]

        with TensorboardWriter(self.model.graph, self.model.tensorboard_log, tb_log_name, self.model.new_tb_log) \
                as writer:
            for t in range(total_timesteps):
                if self.done:
                    break

                obs = []

                for _ in range(self.n_samples):
                    user_input = self.get_input()
                    if user_input is None:
                        break
                    obs.append(user_input)

                if mode == 'b':
                    baseline_proba, estimate = self.baseline(obs)
                    proba = baseline_proba / self.baseline_temp
                elif mode == 'l':
                    proba = self.predict(obs)
                else:
                    proba = self.predict(obs)
                    baseline_proba, estimate = self.baseline(obs)
                    proba = proba + (baseline_proba / self.baseline_temp)

                if not self.env.no_lm:
                    proba = proba + np.log(self.env.lm_proba)

                # Gumbel-max trick for Boltzmann exploration
                if self.boltzmann_exploration:
                    proba += np.random.gumbel(size=proba.shape)
                act = np.argmax(proba)

                # if the user's reward/backspace feedback is incorrect
                error = np.random.random() <= self.error_rate

                _, rew, done, info = self.env.step(act, error)
                target = info['target']

                # log default baseline interface estimates if available
                if mode != 'l' and estimate is not None:
                    default_correct = np.argmax(baseline_proba) == target
                    baseline_estimates[target].append((estimate, default_correct))

                successes += info['correct']

                learn = mode != 'b' and (not disable_learning)
                self.model.step(obs, act, rew, done, info, target, writer=writer, learn=learn)

                if rew == 0:
                    self.env.undo()

                if done:
                    self.env.reset()
                    n_episodes += 1
                else:
                    self.reset_state()

        metrics = {'accuracy': successes / total_timesteps}
        return metrics, baseline_estimates

    def reset(self):
        super(SimulatedUser, self).reset()
        self.reset_state()

    def reset_state(self):
        """
        Resets state of the simulated user. Should be called after each action step in the environment.
        """
        self.sample_index = 0
        self.next_action_index = np.random.randint(self.env.action_space.n)

    def get_next_action_index(self):
        return self.next_action_index
