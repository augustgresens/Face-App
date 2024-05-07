import cv2
import dlib
import numpy as np
from collections import deque
from facial_accessories import FacialAccessories
from pose_estimation import estimate_pose


class FrameProcessor:
    """Class for processing video frames with facial landmark detection and overlays"""

    def __init__(self, detector, predictor, sunglasses, mustache, overlay_img):
        """Initialize the frame processor with the necessary models and overlays"""
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
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.kernel = np.ones((5, 5), np.uint8)
        self.facial_accessories = FacialAccessories()

    def process_frame(self, frame, flags):
        """Process a video frame to apply overlays based on facial landmarks"""
        frame_gray = self.preprocess_frame(frame)
        landmarks = self.update_using_optical_flow(frame_gray)
        success, rotation_vector, translation_vector, detected_landmarks = (
            self.detect_pose(frame_gray)
        )

        if detected_landmarks:
            landmarks = detected_landmarks
            success = True

        landmarks = self.handle_last_good_detection(landmarks, success)

        if landmarks and rotation_vector is not None and translation_vector is not None:
            frame = self.apply_overlays(
                frame, landmarks, flags, rotation_vector, translation_vector
            )

        self.prev_gray = frame_gray.copy()
        if landmarks:
            self.prev_points = np.array(
                [[p.x, p.y] for p in landmarks.parts()], dtype=np.float32
            ).reshape(-1, 1, 2)

        return frame

    def preprocess_frame(self, frame):
        """Preprocess the frame by converting to grayscale and applying filters"""
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = self.clahe.apply(frame_gray)
        frame_gray = cv2.bilateralFilter(frame_gray, 9, 75, 75)
        frame_gray = cv2.morphologyEx(frame_gray, cv2.MORPH_OPEN, self.kernel)
        return frame_gray

    def detect_pose(self, frame_gray):
        """Detect the pose and facial landmarks in a frame"""
        return estimate_pose(
            frame_gray,
            self.detector,
            self.predictor,
            self.get_camera_matrix(frame_gray.shape),
            np.zeros((4, 1)),
        )

    def handle_last_good_detection(self, landmarks, success):
        """Handle cases where detection fails using the last known good detection"""
        if success:
            self.last_good_landmarks = landmarks
            self.frame_history.clear()
        elif self.last_good_landmarks:
            landmarks = self.last_good_landmarks
            self.frame_history.append(landmarks)
            if len(self.frame_history) > 10:
                averaged_landmarks = self.average_landmarks()
                if averaged_landmarks:
                    return averaged_landmarks

        return landmarks

    def average_landmarks(self):
        """Average out the landmarks over the last few frames to smooth transitions"""
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
        """Create a dlib full object detection from averaged landmark positions"""
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

    def update_using_optical_flow(self, frame_gray):
        """Update facial landmarks using optical flow if previous landmarks are known"""
        if self.prev_gray is not None and self.prev_points is not None:
            new_points, st, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, frame_gray, self.prev_points, None, **self.lk_params
            )
            if new_points is not None and st.sum() > len(st) * 0.75:
                self.prev_points = new_points[st == 1].reshape(-1, 1, 2)
                return self.convert_points_to_landmarks(self.prev_points)
        return None

    def convert_points_to_landmarks(self, points):
        """Convert points to dlib full object detection format"""
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

    def apply_overlays(
        self, frame, landmarks, flags, rotation_vector, translation_vector
    ):
        """Apply overlays to the frame based on the given landmarks and flags"""
        camera_matrix = self.get_camera_matrix(frame.shape)
        dist_coeffs = np.zeros((4, 1))
        if flags.get("sunglasses"):
            frame = self.facial_accessories.add_sunglasses(
                frame,
                landmarks,
                self.sunglasses,
                camera_matrix,
                dist_coeffs,
                rotation_vector,
                translation_vector,
            )
        if flags.get("mustache"):
            frame = self.facial_accessories.add_mustache(
                frame,
                self.mustache,
                landmarks,
                camera_matrix,
                dist_coeffs,
                rotation_vector,
                translation_vector,
            )
        if flags.get("overlay"):
            frame = self.facial_accessories.apply_overlay(
                frame,
                landmarks,
                self.overlay_img,
                camera_matrix,
                dist_coeffs,
                rotation_vector,
                translation_vector,
            )
        return frame

    def get_camera_matrix(self, shape):
        """Get the camera matrix based on the frame shape"""
        focal_length = shape[1]
        center = (shape[1] // 2, shape[0] // 2)
        return np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype="double",
        )
