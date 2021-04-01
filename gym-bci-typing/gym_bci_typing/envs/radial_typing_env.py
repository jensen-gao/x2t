import numpy as np
import gym
import random
from gym import spaces
from gym.utils import EzPickle
from abc import ABC


class RadialTypingEnv(gym.Env, EzPickle, ABC):
    """
    Typing environment based on input sequences. For use with the gaze domain.

    :param user: (User) User in the environment.
    :param width: Width of the typing environment. Overwritten if user is a Pygame user.
    :param height: Height of the typing environment. Overwritten if user is a Pygame user.
    """
    def __init__(self, user, n_actions=8, width=1920, height=1080, sentence_length=8):
        EzPickle.__init__(self)
        self.user = user
        self.has_goal = hasattr(self.user, 'goal')
        self.sentence_length = sentence_length

        if hasattr(self.user, 'width') and hasattr(self.user, 'height'):
            self.width = self.user.width
            self.height = self.user.height
        else:
            self.width = width
            self.height = height

        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.user.input_dim)

        self.radius = min(self.width, self.height) / 3
        angles = [(2 * np.pi * k / n_actions) for k in range(n_actions)]
        self.action_coords = np.array([self.radius * np.array([np.cos(angle), np.sin(angle)]) for angle in angles])
        self.targets = self.action_coords / self.radius
        self.n_steps = None

        self.typed = None
        self.curr_actions = None

        self.success = None
        self.can_undo = None
        self.no_lm = True

    def reset(self):
        self.n_steps = 0
        self.typed = []
        self.user.reset()
        self.curr_actions = self._get_actions()
        self.success = []
        self.can_undo = False

    def step(self, action, error=False):
        """
        Selects a word, always types it.
        """
        if self.has_goal:
            assert len(self.typed) < len(self.user.goal)
        else:
            assert len(self.typed) < self.sentence_length

        info = {'is_success': False}

        if not np.isscalar(action):
            action = action.item()

        selected = self.curr_actions[action]
        target = self.user.get_next_action_index()
        correct = int(action == target)
        rew = correct if not error else 1 - correct
        info['target'] = target
        info['correct'] = correct
        self.success.append(action == target)
        self.typed.append(selected)

        if self.has_goal:
            done = len(self.typed) == len(self.user.goal)

        else:
            done = len(self.typed) == self.sentence_length

        if done:
            info['is_success'] = all(self.success)

        else:
            self.curr_actions = self._get_actions()

        self.n_steps += 1
        info['n_steps'] = self.n_steps
        self.can_undo = True
        return None, rew, done, info

    def undo(self):
        """
        If possible, undoes the last typed word.
        """
        assert self.can_undo
        self.can_undo = False
        self.typed = self.typed[:-1]
        self.success = self.success[:-1]
        self.curr_actions = self._get_actions()

    def render(self, mode='human'):
        pass

    def get_input(self):
        """
        Gets an input from the user.
        """
        return self.user.get_input()

    def _get_actions(self):
        """
        Randomizes words in the radial layout, one of which is the correct next word.
        """
        if hasattr(self.user, 'vocab'):
            actions = random.sample(self.user.vocab, self.action_space.n)
        else:
            actions = [''] * self.action_space.n
        if self.has_goal:
            next_word = self.user.goal[len(self.typed)]
            if next_word not in actions:
                actions[-1] = next_word
        random.shuffle(actions)
        return actions

    def center_coord(self, coord):
        """
        Converts Pygame coordinate to a centered coordinate in the typing environment.
        """
        return np.array([coord[0] - self.width / 2, self.height / 2 - coord[1]])

    def uncenter_coord(self, coord):
        """
        Converts centered coordinate in the typing environment to a Pygame coordinate.
        """
        return np.array([coord[0] + self.width / 2, self.height / 2 - coord[1]])
