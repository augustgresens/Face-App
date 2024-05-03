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
        self.last_good_landmarks = None
        self.lk_params = {
            "winSize": (15, 15),
            "maxLevel": 2,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        }
        self.frame_history = deque(maxlen=10)

    def process_frame(self, frame, flags):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)

        landmarks = self.update_using_optical_flow(frame_gray)
        success, rotation_vector, translation_vector, detected_landmarks = (
            estimate_pose(
                frame_gray,
                self.detector,
                self.predictor,
                self.get_camera_matrix(frame.shape),
                np.zeros((4, 1)),
            )
        )

        if detected_landmarks:
            landmarks = detected_landmarks
            success = True

        landmarks = self.handle_last_good_detection(landmarks, success)

        if landmarks:
            frame = self.process_landmarks(
                frame, landmarks, flags, rotation_vector, translation_vector
            )

        self.prev_gray = frame_gray.copy()
        if landmarks:
            self.prev_points = np.array(
                [[p.x, p.y] for p in landmarks.parts()], dtype=np.float32
            ).reshape(-1, 1, 2)

        return frame

    def handle_last_good_detection(self, landmarks, success):
        if success:
            self.last_good_landmarks = landmarks
            self.frame_history.clear()
        elif self.last_good_landmarks and not self.frame_history:
            return self.last_good_landmarks

        if not success:
            self.frame_history.append(landmarks)
            if len(self.frame_history) > 10:
                averaged_landmarks = self.average_landmarks()
                if averaged_landmarks:
                    return averaged_landmarks

        return landmarks

    def average_landmarks(self):
        """Averages out the landmarks over the last few frames to smooth transitions."""
        if not self.frame_history:
            return None

        sum_x = np.zeros((68, 1), dtype=np.float32)
        sum_y = np.zeros((68, 1), dtype=np.float32)
        count = 0

        for frame_landmarks in self.frame_history:
            if frame_landmarks is None:
                continue
            for i in range(68):
                sum_x[i] += frame_landmarks.part(i).x
                sum_y[i] += frame_landmarks.part(i).y
            count += 1

        if count == 0:
            return None

        return self.create_full_object_detection(sum_x, sum_y, count)

    def create_full_object_detection(self, sum_x, sum_y, count):
        points = [
            dlib.point(int(sum_x[i][0] / count), int(sum_y[i][0] / count))
            for i in range(68)
        ]
        rect = dlib.rectangle(
            left=int(min(p.x for p in points)),
            top=int(min(p.y for p in points)),
            right=int(max(p.x for p in points)),
            bottom=int(max(p.y for p in points)),
        )
        return dlib.full_object_detection(rect, points)

    def get_camera_matrix(self, shape):
        focal_length = shape[1]
        center = (shape[1] // 2, shape[0] // 2)
        return np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype="double",
        )

    def update_using_optical_flow(self, frame_gray):
        if self.prev_gray is not None and self.prev_points is not None:
            new_points, st, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, frame_gray, self.prev_points, None, **self.lk_params
            )
            if new_points is not None and st.sum() > 0:
                self.prev_points = new_points[st == 1].reshape(-1, 1, 2)
                return self.convert_points_to_landmarks(self.prev_points)
        return None

    def convert_points_to_landmarks(self, points):
        if points is None or points.size == 0:
            return None

        points = points.reshape(-1, 2)
        rect = dlib.rectangle(
            left=int(min(p[0] for p in points)),
            top=int(min(p[1] for p in points)),
            right=int(max(p[0] for p in points)),
            bottom=int(max(p[1] for p in points)),
        )
        dlib_points = [dlib.point(int(p[0]), int(p[1])) for p in points]
        return dlib.full_object_detection(rect, dlib_points)

    def process_landmarks(
        self, frame, landmarks, flags, rotation_vector, translation_vector
    ):
        camera_matrix = self.get_camera_matrix(frame.shape)
        dist_coeffs = np.zeros((4, 1))
        if flags.get("sunglasses"):
            frame = add_sunglasses(
                frame,
                landmarks,
                self.sunglasses,
                camera_matrix,
                dist_coeffs,
                rotation_vector,
                translation_vector,
            )
        if flags.get("mustache"):
            frame = add_mustache(
                frame,
                self.mustache,
                landmarks,
                camera_matrix,
                dist_coeffs,
                rotation_vector,
                translation_vector,
            )
        if flags.get("overlay"):
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
