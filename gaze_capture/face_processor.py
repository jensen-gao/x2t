import dlib
import cv2
import math
import numpy as np
from gaze_capture.ITrackerData import loadMetadata


class FaceProcessor:
    """
    Processes webcam images, returns features used as input to iTracker model
    """
    def __init__(self, predictor_path):
        self.face_detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.img_dim = 224
        self.face_grid_dim = 25
        self.left_eye_points = [42, 43, 44, 45, 46, 47]
        self.right_eye_points = [36, 37, 38, 39, 40, 41]

        # means from iTracker dataset, used to normalize inputs to iTracker
        self.face_mean = loadMetadata('gaze_capture/mean_face_224.mat', silent=True)['image_mean']
        self.left_eye_mean = loadMetadata('gaze_capture/mean_left_224.mat', silent=True)['image_mean']
        self.right_eye_mean = loadMetadata('gaze_capture/mean_right_224.mat', silent=True)['image_mean']

    def get_gaze_features(self, frame):
        """
        Takes webcam image, returns cropped face, eyes, and binary mask for where face appears in image
        """
        height, width = frame.shape[:2]
        diff = height - width

        # crop image to square because binary mask is square
        if diff > 0:
            frame = frame[math.floor(diff / 2): -math.ceil(diff / 2)]
        elif diff < 0:
            frame = frame[:, -math.floor(diff / 2): math.ceil(diff / 2)]

        gs_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_box = self._get_facial_detections(gs_frame)

        # if no face detections, returns None
        if face_box is None:
            return None

        face = self._get_face(frame, face_box)
        if face is None:
            return None

        face = (face - self.face_mean) / 255

        face_grid = self._get_face_grid(frame, face_box)

        landmarks = self.predictor(gs_frame, face_box)

        og_left_eye = self._get_eye(frame, landmarks, self.left_eye_points)
        og_right_eye = self._get_eye(frame, landmarks, self.right_eye_points)

        left_eye = (og_left_eye - self.left_eye_mean) / 255
        right_eye = (og_right_eye - self.right_eye_mean) / 255

        face = np.moveaxis(face, -1, 0)
        left_eye = np.moveaxis(left_eye, -1, 0)
        right_eye = np.moveaxis(right_eye, -1, 0)
        return face, left_eye, right_eye, face_grid

    def _get_face(self, frame, face_box):
        """
        Takes image, bounding box of face, returns cropped + resized face image in correct format for iTracker
        """
        try:
            face = frame[face_box.top(): face_box.bottom(), face_box.left(): face_box.right()]
            face = cv2.resize(face, (self.img_dim, self.img_dim))
            face = np.flip(face, axis=2)
        except:
            return None
        return face

    def _get_face_grid(self, frame, face_box):
        """
        Takes image, bounding box of face, returns binary mask for where face appears in image
        """
        frame_dim = len(frame)
        top = math.floor(face_box.top() * self.face_grid_dim / frame_dim)
        bottom = math.ceil(face_box.bottom() * self.face_grid_dim / frame_dim)
        left = math.floor(face_box.left() * self.face_grid_dim / frame_dim)
        right = math.ceil(face_box.right() * self.face_grid_dim / frame_dim)
        face_grid = np.zeros((self.face_grid_dim, self.face_grid_dim))
        face_grid[top: bottom, left: right] = 1
        return face_grid

    def _get_eye(self, frame, landmarks, points):
        """
        Takes image, detected landmark locations, landmark indices/points associated with an eye,
        returns cropped eye image
        """
        eye_landmarks = self._get_landmarks(landmarks, points)
        left, top, width, height = cv2.boundingRect(eye_landmarks)

        w_margin = int(width / 3)
        h_margin = (width + 2 * w_margin - height) / 2
        top_margin = math.ceil(h_margin)
        bot_margin = math.floor(h_margin)

        eye = frame[top - top_margin: top + height + bot_margin, left - w_margin: left + width + w_margin]
        eye = cv2.resize(eye, (self.img_dim, self.img_dim))
        eye = np.flip(eye, axis=2)

        return eye

    def _get_facial_detections(self, gs_frame):
        """
        Returns first face detected by facial detector in greyscale image
        """
        detections = self.face_detector(gs_frame)
        if len(detections) == 0:
            return None
        return detections[0]

    @staticmethod
    def _get_landmarks(landmarks, points):
        """
        Takes landmark locations, landmark indices/points, returns locations of those indices/points
        """
        return np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in points])
