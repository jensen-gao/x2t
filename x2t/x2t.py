from functools import partial

import tensorflow as tf
import numpy as np
import gym
import h5py
import random
import math
import os

from stable_baselines import logger
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.deepq.policies import DQNPolicy
from stable_baselines.deepq.replay_buffer import ReplayBuffer
from keras.preprocessing.sequence import pad_sequences
from x2t.build_graph import build_train


class X2T(OffPolicyRLModel):
    """
    Model processes a sequence of inputs, predicts an action.
    Model's output at each timestep is regressed onto the binary reward feedback.

    Code adapted from the Stable Baselines 2 implementation of DQN

    :param policy: (DQNPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, LnMlpPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: (float) learning rate for adam optimizer
    :param buffer_size: (int) number of episodes stored by the replay buffer
    :param batch_size: (int) size of a batched sampled from replay buffer for training
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    :param n_cpu_tf_sess: (int) The number of threads for TensorFlow operations
        If None, the number of cpu of the current machine will be used.
    """

    def __init__(self, policy, env, learning_rate=5e-4, buffer_size=10000, train_freq=1, batch_size=32,
                 learning_starts=0, n_cpu_tf_sess=None, verbose=0, tensorboard_log=None, _init_setup_model=True,
                 policy_kwargs=None, full_tensorboard_log=False, seed=None):

        super(X2T, self).__init__(policy=policy, env=env, replay_buffer=None, verbose=verbose,
                                  policy_base=DQNPolicy, requires_vec_env=False, policy_kwargs=policy_kwargs,
                                  seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)

        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.train_freq = train_freq
        self.learning_rate = learning_rate
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log

        self.graph = None
        self.sess = None
        self._train_step = None
        self.step_model = None
        self.update_target = None
        self.act = None
        self.proba_step = None
        self.replay_buffer = None
        self.step_log = None
        self.params = None
        self.summary = None
        self.callback = None
        self.log_interval = None
        self.new_tb_log = None
        self.episode_rewards = None
        self.episode_successes = None

        if _init_setup_model:
            self.setup_model()

    def _get_pretrain_placeholders(self):
        pass

    def setup_model(self):
        with SetVerbosity(self.verbose):
            assert not isinstance(self.action_space, gym.spaces.Box), \
                "Error: X2T cannot output a gym.spaces.Box action space."
            assert isinstance(self.action_space, gym.spaces.Discrete), \
                "Currently only gym.spaces.Discrete is supported for action space"

            # If the policy is wrap in functool.partial (e.g. to disable dueling)
            # unwrap it to check the class type
            if isinstance(self.policy, partial):
                test_policy = self.policy.func
            else:
                test_policy = self.policy
            assert issubclass(test_policy, DQNPolicy), "Error: the input policy for the X2T model must be " \
                                                       "an instance of DQNPolicy."

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.set_random_seed(self.seed)

                tf_config = tf.ConfigProto(
                    allow_soft_placement=True,
                    inter_op_parallelism_threads=self.n_cpu_tf_sess,
                    intra_op_parallelism_threads=self.n_cpu_tf_sess,
                )
                # Prevent tensorflow from taking all the gpu memory
                tf_config.gpu_options.allow_growth = True

                self.sess = tf.Session(config=tf_config, graph=self.graph)

                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

                self.act, self._train_step, self.step_model, = build_train(
                    q_func=partial(self.policy, **self.policy_kwargs),
                    ob_space=self.observation_space,
                    ac_space=self.action_space,
                    optimizer=optimizer,
                    grad_norm_clipping=10,
                    sess=self.sess,
                    full_tensorboard_log=self.full_tensorboard_log,
                )

                self.proba_step = self.step_model.proba_step
                self.params = tf_util.get_trainable_vars('deepq')

                # Initialize the parameters
                tf_util.initialize(self.sess)
                self.summary = tf.summary.merge_all()

        self.env.user.setup_model(self)

    def learn(self, total_timesteps, callback=None,
              log_interval=100, tb_log_name="run", reset_num_timesteps=True, replay_wrapper=None):
        pass

    def setup_learn(self, callback=None, log_interval=100, reset_num_timesteps=True, replay_wrapper=None):
        """
        Sets up model for learning.
        """

        self.log_interval = log_interval
        self.new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        self.callback = self._init_callback(callback)

        self._setup_learn()

        # Create the replay buffer
        self.replay_buffer = ReplayBuffer(self.buffer_size)

        self.step_log = {'obses': [], 'actions': [], 'rewards': [], 'targets': []}

        self.episode_rewards = [0.0]
        self.episode_successes = []

        self.callback.on_training_start(locals(), globals())
        self.callback.on_rollout_start()

    def step(self, obs, action, rew, done, info, target, writer=None, learn=True):
        """
        Records step to the replay buffer, and performs a learning step from a replay buffer sample if possible.
        """
        with SetVerbosity(self.verbose):
            # Stop training if return value is False
            if self.callback.on_step() is False:
                return

            # Store transition in the replay buffer.
            if learn:
                self.replay_buffer.add(obs, action, rew, None, None)

            self.step_log['obses'].append(obs)
            self.step_log['actions'].append(action)
            self.step_log['targets'].append(target)
            self.step_log['rewards'].append(rew)

            if writer is not None:
                with tf.variable_scope("environment_info", reuse=True):
                    accuracy_summary = tf.Summary(
                        value=[tf.Summary.Value(tag="accuracy", simple_value=1 if target == action else 0)]
                    )
                    writer.add_summary(accuracy_summary, self.num_timesteps)

            self.episode_rewards[-1] += rew

            if done:
                maybe_is_success = info.get('is_success')
                if maybe_is_success is not None:
                    self.episode_successes.append(float(maybe_is_success))
                self.episode_rewards.append(0.0)

                if writer is not None:
                    with tf.variable_scope("environment_info", reuse=True):
                        eps_len_summary = tf.Summary(
                            value=[tf.Summary.Value(tag="episode_length", simple_value=info['n_steps'])])
                        writer.add_summary(eps_len_summary, self.num_timesteps)

            # Do not train if the warmup phase is not over
            if learn and self.num_timesteps > self.learning_starts and self.num_timesteps % self.train_freq == 0 and \
                    len(self.replay_buffer) > 0:
                self.training_step(writer)

            if len(self.episode_rewards[-101:-1]) == 0:
                mean_100ep_reward = -np.inf
            else:
                mean_100ep_reward = round(float(np.mean(self.episode_rewards[-101:-1])), 1)

            num_episodes = len(self.episode_rewards)
            if self.verbose >= 1 and done and self.log_interval is not None and \
                    len(self.episode_rewards) % self.log_interval == 0:
                logger.record_tabular("steps", self.num_timesteps)
                logger.record_tabular("episodes", num_episodes)
                if len(self.episode_successes) > 0:
                    logger.logkv("success rate", np.mean(self.episode_successes[-100:]))
                logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                logger.dump_tabular()

            self.num_timesteps += 1

    # not used
    def predict(self, observation, state=None, mask=None, deterministic=True):
        pass

    # not used
    def action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        pass

    def get_parameter_list(self):
        return self.params

    def save(self, save_path, cloudpickle=False):
        # params
        data = {
            "learning_starts": self.learning_starts,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "train_freq": self.train_freq,
            "verbose": self.verbose,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "policy": self.policy,
            "n_envs": self.n_envs,
            "n_cpu_tf_sess": self.n_cpu_tf_sess,
            "seed": self.seed,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs,
        }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)

    def save_data(self, save_path):
        """
        Saves logged data.
        """
        f = h5py.File(os.path.join(save_path, 'data.hdf5'), 'w')
        for key, val in self.step_log.items():
            f.create_dataset(key, data=np.array(val), compression='gzip')
        f.close()

    def offline_train(self, data, epochs):
        """
        Pretrains model on offline data.
        """
        data = list(zip(data['obses'][()], data['actions'][()], data['rewards'][()]))
        condition = lambda x: x != -1

        data = [step for step in data if condition(step[-1])]

        n_batches = math.ceil(len(data) / self.batch_size)
        with TensorboardWriter(self.graph, self.tensorboard_log, 'offline_log', self.new_tb_log) \
                as writer:
            for i in range(epochs):
                random.shuffle(data)
                for j in range(n_batches):
                    batch = data[j * self.batch_size: (j + 1) * self.batch_size]
                    obs, actions, rews = zip(*batch)
                    obs, actions, rews = np.array(obs), np.array(actions), np.array(rews)
                    # to correct old data format
                    if len(obs.shape) < len(self.env.observation_space.shape) + 2:
                        obs = np.expand_dims(obs, 1)
                    summary = self._train_step(obs, actions, rews, targets=self.env.targets, sess=self.sess)
                    writer.add_summary(summary, i * n_batches + j)

    def training_step(self, writer):
        """
        Samples batch from the replay buffer, takes a gradient step, logs to writer if possible.
        """
        self.callback.on_rollout_end()
        obses, actions, rewards, _, _ = self.replay_buffer.sample(self.batch_size)

        obses = pad_sequences(obses, dtype='float32')
        actions = np.array(actions)
        obses = np.array(obses)

        if writer is not None:
            # run loss backprop with summary, but once every 1000 steps save the metadata
            # (memory, compute time, ...)
            if (1 + self.num_timesteps) % 1000 == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary = self._train_step(obses, actions, rewards, self.env.targets, sess=self.sess,
                                           options=run_options, run_metadata=run_metadata)
                writer.add_run_metadata(run_metadata, 'step%d' % self.num_timesteps)

            else:
                summary = self._train_step(obses, actions, rewards, self.env.targets, sess=self.sess)
            writer.add_summary(summary, self.num_timesteps)
        else:
            _ = self._train_step(obses, actions, rewards, self.env.targets, sess=self.sess)

        self.callback.on_rollout_start()
