import cv2
import numpy as np
from facial_accessories import add_mustache, add_sunglasses, apply_overlay
from pose_estimation import estimate_pose
from draw_axes import draw_axes


class FrameProcessor:
    def __init__(self, detector, predictor, sunglasses, mustache, overlay_img):
        self.detector = detector
        self.predictor = predictor
        self.sunglasses = sunglasses
        self.mustache = mustache
        self.overlay_img = overlay_img

        if self.overlay_img.shape[2] == 3:
            self.overlay_img = cv2.cvtColor(self.overlay_img, cv2.COLOR_RGB2RGBA)

        self.landmark_indices = {
            "forehead": list(range(17, 27)),
            "upper_lip": list(range(48, 60)),
            "left_eye": list(range(36, 42)),
            "right_eye": list(range(42, 48)),
            "nose": list(range(27, 36)),
            "mouth": list(range(60, 68)),
        }

    def process_frame(self, frame, flags):
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
            self.update_facial_points(landmarks)
            frame = self.process_landmarks(frame, landmarks, flags)

        return frame

    def process_landmarks(self, frame, landmarks, flags):
        if flags.get("sunglasses"):
            frame = add_sunglasses(
                frame,
                self.forehead_pts,
                self.left_eye_pts,
                self.right_eye_pts,
                self.sunglasses,
            )
        if flags.get("mustache"):
            frame = add_mustache(
                frame,
                self.upper_lip_pts,
                self.bottom_of_nose_y,
                self.top_of_mouth_y,
                self.mustache,
            )
        if flags.get("overlay"):
            frame = apply_overlay(
                frame,
                landmarks,
                self.overlay_img,
                np.array(
                    [
                        [
                            self.overlay_img.shape[1] * 0.5,
                            self.overlay_img.shape[0] * 0.33,
                        ],
                        [
                            self.overlay_img.shape[1] * 0.5,
                            self.overlay_img.shape[0] * 0.95,
                        ],
                        [
                            self.overlay_img.shape[1] * 0.15,
                            self.overlay_img.shape[0] * 0.25,
                        ],
                        [
                            self.overlay_img.shape[1] * 0.85,
                            self.overlay_img.shape[0] * 0.25,
                        ],
                        [
                            self.overlay_img.shape[1] * 0.3,
                            self.overlay_img.shape[0] * 0.75,
                        ],
                        [
                            self.overlay_img.shape[1] * 0.7,
                            self.overlay_img.shape[0] * 0.75,
                        ],
                        [
                            self.overlay_img.shape[1] * 0.1,
                            self.overlay_img.shape[0] * 0.05,
                        ],
                        [
                            self.overlay_img.shape[1] * 0.9,
                            self.overlay_img.shape[0] * 0.05,
                        ],
                        [
                            self.overlay_img.shape[1] * 0.5,
                            self.overlay_img.shape[0] * 0.5,
                        ],
                    ],
                    dtype="float32",
                ),
                [30, 8, 36, 45, 48, 54, 17, 26, 33],  # Corresponding indices
            )

        return frame

    def update_facial_points(self, landmarks):
        self.forehead_pts = [
            (landmarks.part(i).x, landmarks.part(i).y)
            for i in self.landmark_indices["forehead"]
        ]
        self.upper_lip_pts = [
            (landmarks.part(i).x, landmarks.part(i).y)
            for i in self.landmark_indices["upper_lip"]
        ]
        self.left_eye_pts = [
            (landmarks.part(i).x, landmarks.part(i).y)
            for i in self.landmark_indices["left_eye"]
        ]
        self.right_eye_pts = [
            (landmarks.part(i).x, landmarks.part(i).y)
            for i in self.landmark_indices["right_eye"]
        ]
        self.nose_pts = [
            (landmarks.part(i).x, landmarks.part(i).y)
            for i in self.landmark_indices["nose"]
        ]
        self.mouth_pts = [
            (landmarks.part(i).x, landmarks.part(i).y)
            for i in self.landmark_indices["mouth"]
        ]
        self.bottom_of_nose_y = max(self.nose_pts[6][1], self.nose_pts[7][1])
        self.top_of_mouth_y = min(
            self.mouth_pts[1][1],
            self.mouth_pts[2][1],
            self.mouth_pts[3][1],
            self.mouth_pts[4][1],
        )
