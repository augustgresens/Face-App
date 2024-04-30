import cv2
import dlib
import numpy as np
from collections import deque
from facial_accessories import add_mustache, add_sunglasses, apply_overlay
from pose_estimation import estimate_pose


class FrameProcessor:
    def __init__(self, detector, predictor, sunglasses, mustache, overlay_img):
        self.detector = detector
        self.predictor = predictor
        self.sunglasses = sunglasses
        self.mustache = mustache
        self.overlay_img = overlay_img
        self.prev_gray = None
        self.prev_points = None
        self.lk_params = {
            "winSize": (15, 15),
            "maxLevel": 2,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        }
        self.frame_history = deque(maxlen=10)
        self.kalman_filters = [cv2.KalmanFilter(4, 2) for _ in range(68)]
        for kf in self.kalman_filters:
            kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
            kf.transitionMatrix = np.array(
                [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
            )
            kf.processNoiseCov = 1e-3 * np.eye(4, dtype=np.float32)
            kf.measurementNoiseCov = 1e-1 * np.eye(2, dtype=np.float32)
            kf.errorCovPost = 1e-1 * np.eye(4, dtype=np.float32)
            kf.statePost = np.zeros((4, 1), np.float32)

    def convert_points_to_landmarks(self, points):
        if points is None or points.size == 0:  # Check if points is None or empty
            return None

        dlib_points = [dlib.point(int(p[0]), int(p[1])) for p in points.reshape(-1, 2)]
        if dlib_points:
            min_x = min(p.x for p in dlib_points)
            max_x = max(p.x for p in dlib_points)
            min_y = min(p.y for p in dlib_points)
            max_y = max(p.y for p in dlib_points)
            rect = dlib.rectangle(left=min_x, top=min_y, right=max_x, bottom=max_y)
        else:
            rect = dlib.rectangle(left=0, top=0, right=1, bottom=1)
        return dlib.full_object_detection(rect, dlib_points)

    def process_frame(self, frame, flags):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)
        if self.prev_gray is not None and self.prev_points is not None:
            new_points, st, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, frame_gray, self.prev_points, None, **self.lk_params
            )
            if new_points is not None and st.size == new_points.shape[0]:
                self.prev_points = new_points[st == 1].reshape(-1, 1, 2)

        success, rotation_vector, translation_vector, landmarks = estimate_pose(
            frame_gray,
            self.detector,
            self.predictor,
            self.get_camera_matrix(frame.shape),
            np.zeros((4, 1)),
        )
        if success:
            landmarks = self.update_landmarks_with_kalman(landmarks)
        elif self.prev_points is not None:
            landmarks = self.convert_points_to_landmarks(self.prev_points)

        if landmarks:
            frame = self.process_landmarks(
                frame, landmarks, flags, rotation_vector, translation_vector
            )

        self.prev_gray = frame_gray.copy()
        if success:
            self.prev_points = np.array(
                [[p.x, p.y] for p in landmarks.parts()], dtype=np.float32
            ).reshape(-1, 1, 2)
        return frame

    def get_camera_matrix(self, shape):
        focal_length = shape[1]
        return np.array(
            [
                [focal_length, 0, shape[1] // 2],
                [0, focal_length, shape[0] // 2],
                [0, 0, 1],
            ],
            dtype="double",
        )

    def update_landmarks_with_kalman(self, landmarks):
        for i, point in enumerate(landmarks.parts()):
            kf = self.kalman_filters[i]
            measurement = np.array([[point.x], [point.y]], np.float32)
            kf.correct(measurement)
            prediction = kf.predict()
            landmarks.part(i).x = int(prediction[0])
            landmarks.part(i).y = int(prediction[1])
        return landmarks

    def process_landmarks(
        self, frame, landmarks, flags, rotation_vector, translation_vector
    ):
        camera_matrix = self.get_camera_matrix(frame.shape)
        dist_coeffs = np.zeros((4, 1))
        if (
            flags.get("sunglasses")
            and rotation_vector is not None
            and translation_vector is not None
        ):
            frame = add_sunglasses(
                frame,
                landmarks,
                self.sunglasses,
                camera_matrix,
                dist_coeffs,
                rotation_vector,
                translation_vector,
            )
        if (
            flags.get("mustache")
            and rotation_vector is not None
            and translation_vector is not None
        ):
            frame = add_mustache(
                frame,
                self.mustache,
                landmarks,
                camera_matrix,
                dist_coeffs,
                rotation_vector,
                translation_vector,
            )
        if (
            flags.get("overlay")
            and rotation_vector is not None
            and translation_vector is not None
        ):
            frame = apply_overlay(
                frame,
                landmarks,
                self.overlay_img,
                camera_matrix,
                dist_coeffs,
                rotation_vector,
                translation_vector,
            )
        return frame
