import cv2
import pygame
import os
import h5py
import numpy as np
import torch
import math
from sklearn.svm import LinearSVR
from gym_bci_typing.users import PygameUser
from gaze_capture.face_processor import FaceProcessor
from gaze_capture.ITrackerModel import ITrackerModel


class GazeUser(PygameUser):
    """
    Pygame interface that collects input from the webcam.

    :param predictor_path: (str) Path to the facial feature detector used for segmenting face and eye images.
    :param mode: (str)
    :param calibration_cycles: (int) Number of cycles to collect data during calibration.
    :param calibration_data: (dict) Data for calibration, instead of collecting more.
    :param cam_coord: (Tuple(float)) (x, y) displacement (cm) of the webcam relative to the top left corner of the
    interface window. Used for iTracker gaze estimation.
    :param window_width: (float) Width of the interface window. Used for iTracker gaze estimation.
    :param window_height: (float) Height of the interface window. Used for iTracker gaze estimation.
    """
    def __init__(self, predictor_path, calibration_cycles,
                 cam_coord, window_width, window_height, **kwargs):
        super(GazeUser, self).__init__(input_dim=(128,), **kwargs)
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
        self.cam_coord = np.array(cam_coord)
        self.window_width = window_width
        self.window_height = window_height
        self.calibration_cycles = calibration_cycles
        self.webcam = cv2.VideoCapture(0)

        self.min_x = None
        self.max_x = None
        self.min_y = None
        self.max_y = None
        self.calibration_points = None

    def setup(self, env):
        super(GazeUser, self).setup(env)

        # limits in the normalized centered coordinate system
        self.min_x = -self.width / (2 * self.env.radius)
        self.max_x = -self.min_x
        self.min_y = -self.height / (2 * self.env.radius)
        self.max_y = -self.min_y

        self.calibration_points = self.env.action_coords

    def run(self, total_timesteps, callback=None, tb_log_name='log', mode='m', disable_learning=False):
        self.model.setup_learn(callback=callback)
        if mode in ('b', 'm'):
            signals, gaze_labels = self.record_calibration_data()
            if not self.running:
                exit()

            self.calibrate(signals, gaze_labels)
        self.env.reset()

        if self.running:
            self.main_loop(tb_log_name=tb_log_name, mode=mode, total_timesteps=total_timesteps,
                           disable_learning=disable_learning)
        pygame.quit()

        metrics = {'accuracy': self.successes / self.n_steps}
        return metrics, self.baseline_estimates

    def record_calibration_data(self):
        """
        Runs interface to collect data points for calibration.
        """
        features_list = []
        gaze_labels_list = []

        self.screen.fill(self.bg_color)
        for i, point in enumerate(self.calibration_points):
            color = (255, 255, 255)
            uncentered = self.env.uncenter_coord(point)
            self.draw_circle_with_text(str(i), self.action_text_color, uncentered, self.circle_radius, color)

        self.draw_rect_with_text('Press SPACE to start calibration', self.text_color, self.width,
                                 self.text_field_height, center=self.header_coord)
        pygame.display.flip()

        done = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return None, None
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                        return None, None
                    elif event.key == pygame.K_SPACE:
                        done = True
                        break

        self.draw_rect_with_text("Look at the highlighted number", self.text_color, self.width,
                                 self.text_field_height, center=self.header_coord)

        curr_point = 0
        samples_left = self.n_samples
        last_time = -1001
        cycles = 0
        started = False
        n_points = len(self.calibration_points)

        while self.running and cycles < self.calibration_cycles:
            self.get_event(can_pause=False)

            curr_time = pygame.time.get_ticks()
            if not started:
                prev_point = (curr_point - 1) % n_points
                prev_coord = self.env.uncenter_coord(self.calibration_points[prev_point])
                self.draw_circle_with_text(str(prev_point), self.action_text_color, prev_coord,
                                           self.circle_radius, (255, 255, 255))

                curr_coord = self.env.uncenter_coord(self.calibration_points[curr_point])
                self.draw_circle_with_text(str(curr_point), self.action_text_color, curr_coord,
                                           self.circle_radius, (255, 165, 0))

                pygame.display.flip()
                self.get_event()

                started = True
                samples_left = self.n_samples
                last_time = -1001
                pygame.time.wait(2000)
            else:
                if curr_time > last_time + 100:
                    last_time = curr_time
                    _, frame = self.webcam.read()
                    features = self.face_processor.get_gaze_features(frame)

                    if features is not None:
                        features_list.append(features)
                        gaze_labels_list.append(self.calibration_points[curr_point])
                        samples_left -= 1

                    if samples_left == 0:
                        curr_point += 1
                        if curr_point == len(self.calibration_points):
                            curr_point = 0
                            cycles += 1

                        started = False

        self.screen.fill(self.bg_color)
        self.draw_rect_with_text('Saving and calibrating...', self.text_color, self.width, self.height,
                                 center=(self.width / 2, self.height / 2))
        pygame.display.flip()
        self.get_event()

        gaze_labels = np.array(gaze_labels_list)
        features = zip(*features_list)
        features = [torch.from_numpy(np.array(feature)).float().to(self.device) for feature in features]

        batch_size = 32
        n_batches = math.ceil(len(features_list) / batch_size)
        signals = []
        for i in range(n_batches):
            batch = [feature[i * batch_size: (i + 1) * batch_size] for feature in features]
            output = self.i_tracker(*batch)
            signals.extend(output.detach().cpu().numpy())

        signals = np.array(signals)

        f = h5py.File(os.path.join(self.model.tensorboard_log, 'calibration_data.hdf5'), 'w')
        f.attrs['width'] = self.width
        f.attrs['height'] = self.height
        f.attrs['window_width'] = self.window_width
        f.attrs['window_height'] = self.window_height
        f.attrs['cam_coord'] = self.cam_coord
        f.create_dataset('signals', data=signals, compression='gzip')
        f.create_dataset('gaze_labels', data=gaze_labels, compression='gzip')
        f.close()

        return signals, gaze_labels

    def calibrate(self, signals, gaze_labels):
        """
        Fits a linear SVR gaze estimation model from the final hidden layer of
        iTracker on the collected data points.
        """
        #  convert from unnormalized env coordinate system to relative displacement from camera
        gaze_labels = gaze_labels * np.array([[self.window_width / self.width, self.window_height / self.height]])
        gaze_labels = gaze_labels - self.cam_coord[None]

        x_labels, y_labels = zip(*gaze_labels)
        self.x_svr_gaze_estimator.fit(signals, x_labels)
        self.y_svr_gaze_estimator.fit(signals, y_labels)

    def baseline(self, obs):
        """
        Outputs the default baseline interface's distribution over actions, by taking the 2D gaze estimates from the
        calibrated linear SVR model, averaging them over all samples, and calculating the negative distances from
        the average estimate to each of the action coordinates.
        """
        estimates = np.stack((self.x_svr_gaze_estimator.predict(obs), self.y_svr_gaze_estimator.predict(obs)), axis=1)

        # convert from relative displacement from camera to normalized env coordinate system
        gaze_coords = (estimates + self.cam_coord[None]) * np.array(
            [[self.width / (self.env.radius * self.window_width),
              self.height / (self.env.radius * self.window_height)]])
        clipped = self.clip_coords(gaze_coords)
        average = np.mean(clipped, axis=0)
        distances = np.linalg.norm((self.env.action_coords / self.env.radius) - average, axis=1)
        return -distances, average

    def clip_coords(self, coords):
        """
        Clip coordinate to the interface's limits.
        """
        coords[:, 0] = np.clip(coords[:, 0], self.min_x, self.max_x)
        coords[:, 1] = np.clip(coords[:, 1], self.min_y, self.max_y)
        return coords

    def get_input(self):
        """
        Gets image from the webcam, processes it to obtain the features used as input to iTracker, then runs iTracker
        on these features to get processed gaze features.
        """
        _, frame = self.webcam.read()
        features = self.face_processor.get_gaze_features(frame)

        # returns zero signal if no face detected
        if features is None:
            return np.zeros(self.input_dim)
        else:
            i_tracker_input = [torch.from_numpy(feature)[None].float().to(self.device) for feature in features]
            i_tracker_features = self.i_tracker(*i_tracker_input).detach().cpu().numpy()
            return i_tracker_features[0]

    def reset_actions(self, log_proba=None):
        self.actions = self.env.curr_actions

        for i, action in enumerate(self.actions):
            coord = self.env.uncenter_coord(self.env.action_coords[i])
            color = (255, 255, 255)
            text_color = self.action_text_color
            self.draw_rect_with_text(action, text_color, self.button_width, self.button_height, color, center=coord)
