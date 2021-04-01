import numpy as np
import gym
from gym import spaces
from gym.utils import EzPickle
from abc import ABC
from transformer_xl.tf.lm.word_lm import WordLM
import string
import itertools


class CharTypingEnv(gym.Env, EzPickle, ABC):
    """
    Typing environment based on input sequences. For use with in the simulated handwriting domain.

    :param user: (User) User in the environment.
    :param width: Width of the typing environment. Overwritten if user is a Pygame user.
    :param height: Height of the typing environment. Overwritten if user is a Pygame user.
    """
    def __init__(self, user, width=1920, height=1080, temperature=1, smoothing=0.001, no_lm=False):
        EzPickle.__init__(self)
        self.user = user
        self.temperature = temperature
        self.smoothing = smoothing

        if hasattr(self.user, 'width') and hasattr(self.user, 'height'):
            self.width = self.user.width
            self.height = self.user.height
        else:
            self.width = width
            self.height = height

        self.actions = [' '] + list(string.ascii_lowercase)
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.user.input_dim)

        # Targets not used in this environment
        self.targets = np.zeros((self.action_space.n, 2))

        self.n_steps = None
        self.typed = None

        self.no_lm = no_lm
        if not self.no_lm:
            self.lm_proba = None
            self.lm = WordLM()
            self.logits = None
            self.init_logits = self.lm.get_logits('') / self.temperature
            self.init_lm_proba = self._compute_lm_proba(self.init_logits)

        self.success = None

    def reset(self):
        self.n_steps = 0
        self.typed = []
        if not self.no_lm:
            self.logits = self.init_logits
            self.lm_proba = self.init_lm_proba
        self.user.reset()
        self.success = []

    def step(self, action, error=False):
        """
        Selects a character, but only types it if the simulated reward/backspace feedback is positive.
        """
        typed_string = ''.join(self.typed)
        assert len(typed_string) < len(self.user.goal)

        done = False
        info = {'is_success': False}

        if not np.isscalar(action):
            action = action.item()

        selected = self.actions[action]
        target = self.user.get_next_action_index()
        correct = int(action == target)

        # reward/backspace feedback is correct if there is no error
        rew = correct if not error else 1 - correct
        info['target'] = target
        info['correct'] = correct

        # only types character if reward is positive (no backspace)
        if rew > 0:
            self.success.append(correct)

            if (not self.typed) or self.typed[-1] == ' ' or selected == ' ':
                self.typed.append(selected)
            else:
                self.typed[-1] += selected

            typed_string += selected
            if len(typed_string) == len(self.user.goal):
                done = True
                info['is_success'] = typed_string == self.user.goal

            elif not self.no_lm:
                if selected == ' ':
                    self.logits = self.lm.get_logits(typed_string.strip()) / self.temperature

                self.lm_proba = self._compute_lm_proba(self.logits)

        self.n_steps += 1
        info['n_steps'] = self.n_steps
        return None, rew, done, info

    def render(self, mode='human'):
        pass

    def get_input(self):
        """
        Gets an input from the user.
        """
        return self.user.get_input()

    def _compute_lm_proba(self, logits, eps=1e-6):
        """
        Converts language model word logits into probability distribution over characters/actions, by using a trie
        to combine logits of words that have the same next letter. If in the middle of a word, filters logits to
        only have words consistent with the prefix of the word typed so far.
        """
        assert not self.no_lm
        if (not self.typed) or self.typed[-1] == ' ':
            prefix = ''
        else:
            prefix = self.typed[-1]
            if not self.lm.lm_vocab.vocab_trie.has_node(prefix):
                return np.ones(self.action_space.n) / self.action_space.n
        proba = np.full(self.action_space.n, eps)

        for i, action in enumerate(self.actions):
            if action == ' ':
                try:
                    indices = self.lm.lm_vocab.vocab_trie.__getitem__(prefix)
                    proba[i] += np.sum(np.exp(logits[indices]))
                except KeyError:
                    pass
            elif self.lm.lm_vocab.vocab_trie.has_node(prefix + action):
                indices = list(itertools.chain.from_iterable(
                    self.lm.lm_vocab.vocab_trie.values(prefix + action)))
                proba[i] += np.sum(np.exp(logits[indices]))

        total_proba = np.sum(proba)

        # smoothing, to give nonzero probabilities to all characters/actions
        smoothing = self.smoothing * total_proba
        proba += smoothing
        proba /= total_proba + self.action_space.n * smoothing
        return proba

    # not needed for this env
    def undo(self):
        pass
