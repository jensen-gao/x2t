from sklearn.svm import LinearSVR
from gaze_capture.face_processor import FaceProcessor
from gaze_capture.ITrackerModel import ITrackerModel
from gym_bci_typing.users.simulated_users.simulated_user import SimulatedUser
import torch
import numpy as np


class SimulatedOfflineGaze(SimulatedUser):
    """
    Version of the gaze user, but instead of real gaze inputs, replays gaze data collected from past real gaze
    sessions. Automates backspace feedback by making giving a random chance of being incorrect.
    """
    def __init__(self, data, calibration_data, predictor_path, **kwargs):
        super(SimulatedOfflineGaze, self).__init__(input_dim=(128,), **kwargs)
        self.i_tracker = ITrackerModel()
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.i_tracker.cuda()
            state = torch.load('gaze_capture/checkpoint.pth.tar')['state_dict']
        else:
            self.device = "cpu"
            state = torch.load('gaze_capture/checkpoint.pth.tar', map_location=torch.device('cpu'))['state_dict']
        self.i_tracker.load_state_dict(state, strict=False)
        self.x_svr_gaze_estimator = LinearSVR(max_iter=5000)
        self.y_svr_gaze_estimator = LinearSVR(max_iter=5000)

        self.face_processor = FaceProcessor(predictor_path)
        self.traj_index = None

        self.data = data
        self.curr_trajectory = None

        # load calibration data and attributes if available
        if calibration_data is not None:
            self.window_width = calibration_data.attrs['window_width']
            self.window_height = calibration_data.attrs['window_height']
            self.width = calibration_data.attrs['width']
            self.height = calibration_data.attrs['height']
            self.cam_coord = np.array(calibration_data.attrs['cam_coord'])
            signals, gaze_labels = calibration_data['signals'][()], calibration_data['gaze_labels'][()]
            self.calibrate(signals, gaze_labels)

            self.min_x = None
            self.max_x = None
            self.min_y = None
            self.max_y = None

    def get_input(self):
        if self.sample_index >= len(self.curr_trajectory):
            return None
        user_input = self.curr_trajectory[self.sample_index]
        self.sample_index += 1
        return user_input

    def reset_state(self):
        self.sample_index = 0
        if self.traj_index >= len(self.data['obses'][()]):
            self.done = True
            return

        self.curr_trajectory = self.data['obses'][()][self.traj_index]
        self.next_action_index = self.data['targets'][()][self.traj_index]
        self.traj_index += 1

    def baseline(self, obs):
        estimates = np.stack((self.x_svr_gaze_estimator.predict(obs), self.y_svr_gaze_estimator.predict(obs)), axis=1)

        # convert from relative displacement from camera to normalized env coordinate system
        gaze_coords = (estimates + self.cam_coord[None]) * np.array(
            [[self.width / (self.env.radius * self.window_width),
              self.height / (self.env.radius * self.window_height)]])
        clipped = self.clip_coords(gaze_coords)
        average = np.mean(clipped, axis=0)
        distances = np.linalg.norm(self.env.targets - average, axis=1)
        return -distances, average

    def clip_coords(self, coords):
        """
        Clip coordinate to the interface's limits.
        """
        coords[:, 0] = np.clip(coords[:, 0], self.min_x, self.max_x)
        coords[:, 1] = np.clip(coords[:, 1], self.min_y, self.max_y)
        return coords

    def setup(self, env):
        super(SimulatedOfflineGaze, self).setup(env)

        if hasattr(self, 'min_x'):
            self.min_x = -self.width / (2 * self.env.radius)
            self.max_x = -self.min_x
            self.min_y = -self.height / (2 * self.env.radius)
            self.max_y = -self.min_y

        self.traj_index = 0

    def calibrate(self, signals, gaze_labels):
        # convert from unnormalized env coordinate system to relative displacement from camera
        gaze_labels = gaze_labels * np.array([[self.window_width / self.width, self.window_height / self.height]])
        gaze_labels = gaze_labels - self.cam_coord[None]

        x_labels, y_labels = zip(*gaze_labels)
        self.x_svr_gaze_estimator.fit(signals, x_labels)
        self.y_svr_gaze_estimator.fit(signals, y_labels)

    def run(self, total_timesteps=None, callback=None, tb_log_name='log', mode='m', disable_learning=False):
        total_timesteps = len(self.data['obses'])
        return super(SimulatedOfflineGaze, self).run(total_timesteps, callback, tb_log_name, mode, disable_learning)
