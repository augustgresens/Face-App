import cv2
import numpy as np
from facial_accessories import add_mustache, add_sunglasses
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

            if flags["overlay"]:
                frame = self.apply_overlay(
                    frame,
                    rotation_vector,
                    translation_vector,
                    camera_matrix,
                    dist_coeffs,
                )

        return frame

    def apply_overlay(
        self, frame, rotation_vector, translation_vector, camera_matrix, dist_coeffs
    ):
        angle = np.linalg.norm(rotation_vector) * (180 / np.pi)
        y_rotation = rotation_vector[1]

        model_points_3d = np.array(
            [
                (0.0, 0.0, 0.0),  # Nose tip
                (-300.0, -300.0, -125.0),  # Left cheek
                (300.0, -300.0, -125.0),  # Right cheek
                (0.0, 300.0, -125.0),  # Chin
            ],
            dtype="double",
        )

        image_points_2d, _ = cv2.projectPoints(
            model_points_3d,
            rotation_vector,
            translation_vector,
            camera_matrix,
            dist_coeffs,
        )

        image_points_2d = image_points_2d.reshape(-1, 2).astype(np.float32)

        width_factor = max(0.5, 1 - abs(y_rotation) / 2)
        half_width = int(self.overlay_img.shape[1] / 2 * width_factor)
        half_height = int(self.overlay_img.shape[0] / 2)

        overlay_points = np.array(
            [
                [self.overlay_img.shape[1] / 2, half_height],
                [
                    self.overlay_img.shape[1] / 2 - half_width,
                    self.overlay_img.shape[0],
                ],
                [
                    self.overlay_img.shape[1] / 2 + half_width,
                    self.overlay_img.shape[0],
                ],
                [self.overlay_img.shape[1] / 2, 0],
            ],
            dtype=np.float32,
        )

        matrix = cv2.getPerspectiveTransform(overlay_points, image_points_2d)

        transformed_overlay = cv2.warpPerspective(
            self.overlay_img, matrix, (frame.shape[1], frame.shape[0])
        )

        if transformed_overlay.shape[2] == 3:
            transformed_overlay = cv2.cvtColor(transformed_overlay, cv2.COLOR_BGR2BGRA)

        alpha_overlay = transformed_overlay[:, :, 3] / 255.0
        alpha_frame = 1.0 - alpha_overlay
        overlay_color = transformed_overlay[:, :, :3]

        for c in range(3):
            frame[:, :, c] = (
                alpha_overlay * overlay_color[:, :, c] + alpha_frame * frame[:, :, c]
            )

        return frame

    def process_landmarks(self, frame, landmarks, flags):
        self.update_facial_points(landmarks)

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
