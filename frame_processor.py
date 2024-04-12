import cv2
import numpy as np
from facial_accessories import add_mustache, add_sunglasses
from pose_estimation import estimate_pose
from draw_axes import draw_axes


class FrameProcessor:
    def __init__(self, detector, predictor, sunglasses, mustache):
        self.detector = detector
        self.predictor = predictor
        self.sunglasses = sunglasses
        self.mustache = mustache
        # Initialize empty lists for facial points that will be updated per frame
        self.forehead_pts = []
        self.upper_lip_pts = []
        self.left_eye_pts = []
        self.right_eye_pts = []
        self.nose_pts = []
        self.mouth_pts = []
        self.bottom_of_nose_y = 0
        self.top_of_mouth_y = 0

    def process_frame(self, frame, flags):
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(grayscale)

        size = frame.shape
        focal_length = size[1]
        center = (size[1] // 2, size[0] // 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype="double",
        )
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, translation_vector, landmarks = estimate_pose(
            frame, self.detector, self.predictor, camera_matrix, dist_coeffs
        )

        if success:
            if flags["axes"]:
                frame = draw_axes(
                    frame,
                    rotation_vector,
                    translation_vector,
                    camera_matrix,
                    dist_coeffs,
                )
            if landmarks:
                frame = self.process_landmarks(frame, landmarks, flags)

        return frame

    def process_landmarks(self, frame, landmarks, flags):
        # Update facial points based on detected landmarks
        self.update_facial_points(landmarks)

        # Apply accessories if flags are set
        if flags["sunglasses"]:
            frame = add_sunglasses(
                frame,
                self.forehead_pts,
                self.left_eye_pts,
                self.right_eye_pts,
                self.sunglasses,
            )
        if flags["mustache"]:
            frame = add_mustache(
                frame,
                self.upper_lip_pts,
                self.bottom_of_nose_y,
                self.top_of_mouth_y,
                self.mustache,
            )

        return frame

    def update_facial_points(self, landmarks):
        self.forehead_pts = [
            (landmarks.part(i).x, landmarks.part(i).y) for i in range(17, 27)
        ]
        self.upper_lip_pts = [
            (landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 60)
        ]
        self.left_eye_pts = [
            (landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)
        ]
        self.right_eye_pts = [
            (landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)
        ]
        self.nose_pts = [
            (landmarks.part(i).x, landmarks.part(i).y) for i in range(27, 36)
        ]
        self.mouth_pts = [
            (landmarks.part(i).x, landmarks.part(i).y) for i in range(60, 68)
        ]
        self.bottom_of_nose_y = max(self.nose_pts[6][1], self.nose_pts[7][1])
        self.top_of_mouth_y = min(
            self.mouth_pts[1][1],
            self.mouth_pts[2][1],
            self.mouth_pts[3][1],
            self.mouth_pts[4][1],
        )
