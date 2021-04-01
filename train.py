import numpy as np
import tensorflow as tf
import random
import gym
import argparse
import re
import json
import os
import pickle
import h5py
from shutil import copyfile
from datetime import datetime

# Script used for all X2T experiments

parser = argparse.ArgumentParser()
parser.add_argument('--load_path', '-l', type=str, default=None,
                    help='Path to load model parameters from.')
parser.add_argument('--save', '-s', type=str, default=None,
                    help='Folder name to save to. If None, defaults to the current date and time.')
parser.add_argument('--data_path', '-dp', type=str, default=None,
                    help='Path to load offline data from for pretraining. Should be a data.hdf5 file previously produced'
                         'by this script.')
parser.add_argument('--n_steps', '-n', type=int, default=1000,
                    help='Number of steps to run the session for.')
parser.add_argument('--learning_rate', '-lr', type=float, default=None,
                    help='Learning rate for each gradient step.')
parser.add_argument('--batch_size', '-bs', type=int, default=128,
                    help='Minibatch sample size from replay buffer.')
parser.add_argument('--n_samples', '-sa', type=int, default=10,
                    help='Number of input samples per action step. Overwritten to 1 if using sim UJI.')
parser.add_argument('--user', '-u', type=str, default='su',
                    help='[rg|og|su] Which type of user. Real gaze [rg], offline gaze [og], sim UJI [su]')
parser.add_argument('--seed', '-sd', type=int, default=0, help='Random seed.')
parser.add_argument('--buffer_size', '-bfs', type=int, default=500,
                    help='Size of the replay buffer.')
parser.add_argument('--mode', '-m', type=str, default='m',
                    help='[b|l|m] Which prediction mode to use. Default baseline only [b], learned model only [l], or '
                         'default baseline mixed with learned model [m].')
parser.add_argument('--keep_prob', '-kp', type=float, default=None,
                    help='Keep probability for dropout.')
parser.add_argument('--epochs', '-e', type=int, default=None,
                    help='Number of epochs for offline pretraining.')
parser.add_argument('--offline_only', '-oo', action='store_true',
                    help='Whether to only train a model on offline data. Requires data_path to be set.')
parser.add_argument('--baseline_temp', '-bt', type=float, default=1,
                    help='Temperature for the default baseline predictions.')
parser.add_argument('--use_recent', '-ur', action='store_true',
                    help='Whether to use the most recent default baseline data to use for pretraining.'
                         'Overwrites data_path.')
parser.add_argument('--no_recent', '-nr', action='store_true',
                    help='Whether to not overwrite the most recent default baseline data.')
parser.add_argument('--disable_learning', '-dl', action='store_true',
                    help='Disable learning.')
parser.add_argument('--boltzmann_exploration', '-be', action='store_true',
                    help='Whether to choose actions using Boltzmann exploration instead of argmax.')
parser.add_argument('--learning_starts', '-ls', type=int, default=None,
                    help='Which timestep to start learning.')

# for simulated users only
parser.add_argument('--error_rate', '-er', type=float, default=0,
                    help='Error rate for the binary feedback system of the simulated user.')

# for gaze only
parser.add_argument('--units', '-un', type=int, default=64,
                    help='Number of units in the single layer network used as the reward model for gaze.')
parser.add_argument('--n_actions', '-na', type=int, default=8,
                    help='Number of actions in the radial environment. If using an offline gaze user, must be less'
                         'than or equal to the number of actions in the environment the data was collected in.')

# for offline gaze only
parser.add_argument('--offline_gaze_path', '-gp', type=str, default=None,
                    help='Path to load data and calibration data for the offline gaze user. Should be a folder'
                         'containing data.hdf5 and calibration.hdf5 files previously produced by this script.')

# for real gaze only
parser.add_argument('--sampling_freq', '-sf', type=int, default=10,
                    help='Input sampling frequency for the real users (Hz).')
parser.add_argument('--windowed', '-w', action='store_true',
                    help='Whether to run the interface in windowed mode, if using a real user.'
                         'Useful for debugging, otherwise should run in default fullscreen.')
parser.add_argument('--calibration_cycles', '-cc', type=int, default=2,
                    help='Number of cycles for calibrating the real gaze user.')
parser.add_argument('--cam_x', '-cx', type=float, default=14.5,
                    help='Horizontal displacement (cm) of the camera from the left end of the window for the real'
                         'gaze user. Positive values indicate rightward displacement.'
                         'Default value for a 13 inch MacBook Pro in fullscreen.')
parser.add_argument('--cam_y', '-cy', type=float, default=-0.5,
                    help='Vertical displacement (cm) of the camera from the top of the window for the real'
                         'gaze user. Positive values indicate downward displacement.'
                         'Default value for a 13 inch MacBook Pro in fullscreen.')
parser.add_argument('--window_width', '-ww', type=float, default=29,
                    help='Width (cm) of the window for the real gaze user. Default value for a 13 inch MacBook'
                         'Pro in fullscreen.')
parser.add_argument('--window_height', '-wh', type=float, default=18,
                    help='Height (cm) of the window for the real gaze user. Default value for a 13 inch MacBook'
                         'Pro in fullscreen.')

# for simulated UJI only
parser.add_argument('--user_index', '-ui', type=int, default=0,
                    help='Which writer\'s data to use for the sim UJI user.'
                         'Valid indices from 0-59 inclusive.')
parser.add_argument('--drift_seed', '-ds', type=int, default=0, help='Random drift seed.')
parser.add_argument('--drift_std', '-dst', type=float, default=0.0002,
                    help='Standard deviation of Gaussian noise for drift. Units are in the normalized'
                         'coordinate system where the distance from the center to any action is 1 unit.')
parser.add_argument('--lm', '-lm', action='store_true',
                    help='Whether to use a LM for a prior.')

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
tf.set_random_seed(args.seed)

from gym_bci_typing.users import *
from x2t import X2T
from x2t.policies import TypingPolicy


def run(args):
    if args.n_steps is None:
        args.n_steps = 1000 if args.user == 'su' else 300

    if args.learning_rate is None:
        args.learning_rate = 5e-4 if args.user == 'su' else 1e-3

    if args.keep_prob is None:
        args.keep_prob = 0.5 if args.user == 'su' else 0.7

    if args.epochs is None:
        args.epochs = 20 if args.user == 'su' else 50

    if args.learning_starts is None:
        args.learning_starts = 100 if args.user == 'su' else 4

    folder_names = {'og': 'offline_gaze', 'rg': 'real_gaze', 'su': 'sim_uji'}

    user_save_path = os.path.join('./experiments/', folder_names[args.user])
    if args.save is None:
        time = datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
        save_path = os.path.join(user_save_path, time)
    else:
        save_path = os.path.join(user_save_path, args.save)
    os.makedirs(save_path)

    if args.use_recent:
        offline_data = h5py.File(os.path.join(user_save_path, 'recent/baseline_data.hdf5'))
    elif args.data_path is not None:
        offline_data = h5py.File(args.data_path, 'r')
    else:
        offline_data = None

    with open(os.path.join(save_path, 'params.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    user = create_user(args, save_path)
    env = create_env(args, user)
    user.setup(env)
    model = create_model(args, env, save_path)

    if args.offline_only:
        assert offline_data is not None

    if offline_data is not None:
        model.offline_train(offline_data, epochs=args.epochs)
        model.save(os.path.join(save_path, 'offline_model'))

    if not args.offline_only:
        metrics, baseline_estimates = user.run(total_timesteps=args.n_steps, mode=args.mode,
                                               disable_learning=args.disable_learning)

        if sum([len(x) for x in baseline_estimates]) > 0:
            with open(os.path.join(save_path, 'baseline_estimates.pkl'), 'wb') as f:
                pickle.dump(baseline_estimates, f)

        with open(os.path.join(save_path, 'metrics.txt'), 'w') as f:
            lines = [k + ': ' + str(v) + '\n' for k, v in metrics.items()]
            f.writelines(lines)
            model.save_data(save_path)

    if args.mode != 'b':
        model.save(os.path.join(save_path, 'final_model'))

    data_path = os.path.join(save_path, 'data.hdf5')

    if os.path.isfile(data_path) and args.mode == 'b' and not args.no_recent:
        recent_save_path = os.path.join(user_save_path, 'recent')
        os.makedirs(recent_save_path, exist_ok=True)
        copyfile(data_path, os.path.join(recent_save_path, 'baseline_data.hdf5'))


def create_user(args, save_path):
    """
    Creates user object.
    """
    if args.user == 'og':
        assert args.offline_gaze_path is not None
        gaze_data_path = os.path.join(args.offline_gaze_path, 'data.hdf5')
        offline_gaze_data = h5py.File(gaze_data_path, 'r')

        if not args.mode == 'l':
            calibration_data_path = os.path.join(args.offline_gaze_path, 'calibration_data.hdf5')
            calibration_data = h5py.File(calibration_data_path, 'r')
        else:
            calibration_data = None

        user = SimulatedOfflineGaze(data=offline_gaze_data, calibration_data=calibration_data,
                                    predictor_path='./data/shape_predictor_68_face_landmarks.dat',
                                    error_rate=args.error_rate / 10, n_samples=args.n_samples,
                                    baseline_temp=args.baseline_temp, boltzmann_exploration=args.boltzmann_exploration)
    else:
        with open('./data/mocha-timit.txt', 'r') as f:
            goals = f.readlines()

        # preprocessing specific to MOCHA-TIMIT text file format
        goals = [goal for goal in goals if goal.strip()]
        pattern = re.compile(r'[^a-zA-Z\s]+', re.UNICODE)
        goals = [pattern.sub('', line).strip().lower().split() for line in goals]

        if args.user == 'rg':
            user = GazeUser(goals=goals, sampling_freq=args.sampling_freq,
                            save_path=save_path, predictor_path='./data/shape_predictor_68_face_landmarks.dat',
                            calibration_cycles=args.calibration_cycles, windowed=args.windowed,
                            cam_coord=(args.cam_x, args.cam_y), window_width=args.window_width,
                            window_height=args.window_height, n_samples=args.n_samples,
                            baseline_temp=args.baseline_temp, boltzmann_exploration=args.boltzmann_exploration)

        elif args.user == 'su':
            with open('./data/online_handwriting.pkl', 'rb') as f:
                data = pickle.load(f)
            for key in data.keys():
                # select only pen strokes for the desired writer/user index
                data[key] = data[key][2 * args.user_index:2 * (args.user_index + 1)]

            user = SimulatedUJI(goals=goals, data=data, drift_std=args.drift_std,
                                error_rate=args.error_rate, drift_seed=args.drift_seed,
                                baseline_temp=args.baseline_temp, boltzmann_exploration=args.boltzmann_exploration)
        else:
            raise NotImplementedError

    return user


def create_env(args, user):
    """
    Creates typing environment.
    """
    if args.user == 'su':
        env = gym.make('gym_bci_typing:CharTyping-v0', user=user, no_lm=not args.lm)
    else:
        env = gym.make('gym_bci_typing:RadialTyping-v0', user=user, n_actions=args.n_actions)
    return env


def create_model(args, env, save_path):
    """
    Creates the X2T model.
    """
    policy_kwargs = {'units': args.units, 'mode': 'char' if args.user == 'su' else 'gaze',
                     'keep_prob': args.keep_prob}
    policy = TypingPolicy

    model = X2T(policy, env, learning_rate=args.learning_rate, batch_size=args.batch_size,
                learning_starts=args.learning_starts, verbose=1, tensorboard_log=save_path,
                policy_kwargs=policy_kwargs, buffer_size=args.buffer_size)

    if args.load_path is not None:
        model.load_parameters(args.load_path, exact_match=False)

    return model


run(args)
