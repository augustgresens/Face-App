import cv2
import numpy as np


class FacialAccessories:
    """Class to manage facial accessories overlays such as sunglasses and mustaches"""

    def alpha_blend(self, frame, overlay, position=(0, 0)):
        """Blend an overlay image with transparency onto a background frame at a specified position"""
        x, y = position
        overlay_height, overlay_width = overlay.shape[:2]

        x_end = min(x + overlay_width, frame.shape[1])
        y_end = min(y + overlay_height, frame.shape[0])
        x = max(x, 0)
        y = max(y, 0)

        roi = frame[y:y_end, x:x_end]
        overlay_roi = overlay[0 : (y_end - y), 0 : (x_end - x)]

        alpha_overlay = overlay_roi[:, :, 3] / 255.0
        alpha_frame = 1.0 - alpha_overlay
        for c in range(3):
            roi[:, :, c] = (
                alpha_overlay * overlay_roi[:, :, c] + alpha_frame * roi[:, :, c]
            ).astype(np.uint8)

        frame[y:y_end, x:x_end] = roi
        return frame

    def compute_homography(self, src_points, dst_points):
        """Compute the homography transformation matrix"""
        transformation_matrix, _ = cv2.findHomography(src_points, dst_points)
        return transformation_matrix

    def transform_overlay(self, overlay, transformation_matrix, frame_dimensions):
        """Transform an overlay using a given transformation matrix"""
        return cv2.warpPerspective(
            overlay, transformation_matrix, (frame_dimensions[1], frame_dimensions[0])
        )

    def add_mustache(
        self,
        frame,
        mustache,
        landmarks,
        camera_matrix,
        dist_coeffs,
        rotation_vector,
        translation_vector,
    ):
        """Add a mustache overlay to a frame based on facial landmarks"""
        scale_factor = 0.45
        resized_mustache = cv2.resize(
            mustache,
            None,
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_LINEAR,
        )

        _, pitch, _ = cv2.Rodrigues(rotation_vector)[0]
        pitch = pitch[0]

        vertical_offset = -50 + int(20 * pitch)

        mustache_3d_points = np.array(
            [
                [-resized_mustache.shape[1] / 2, vertical_offset, 0],
                [resized_mustache.shape[1] / 2, vertical_offset, 0],
                [
                    -resized_mustache.shape[1] / 2,
                    vertical_offset - resized_mustache.shape[0],
                    0,
                ],
                [
                    resized_mustache.shape[1] / 2,
                    vertical_offset - resized_mustache.shape[0],
                    0,
                ],
            ]
        )

        image_points_2d, _ = cv2.projectPoints(
            mustache_3d_points,
            -rotation_vector,
            translation_vector,
            camera_matrix,
            dist_coeffs,
        )

        dst_points = image_points_2d.reshape(-1, 2)
        src_points = np.array(
            [
                [0, 0],
                [resized_mustache.shape[1], 0],
                [0, resized_mustache.shape[0]],
                [resized_mustache.shape[1], resized_mustache.shape[0]],
            ],
            dtype="float32",
        )

        transformation_matrix = self.compute_homography(src_points, dst_points)
        if transformation_matrix is not None:
            transformed_mustache = self.transform_overlay(
                resized_mustache, transformation_matrix, frame.shape
            )
            frame = self.alpha_blend(frame, transformed_mustache)

        return frame

    def add_sunglasses(
        self,
        frame,
        landmarks,
        sunglasses,
        camera_matrix,
        dist_coeffs,
        rotation_vector,
        translation_vector,
    ):
        """Add a sunglasses overlay to a frame based on facial landmarks"""
        scale_factor = 1.7
        corrected_rotation_vector = -rotation_vector.copy()
        corrected_rotation_vector[2] *= 0.1

        _, pitch, _ = cv2.Rodrigues(corrected_rotation_vector)[0]
        pitch = pitch[0]
        vertical_offset = int(275 + 20 * pitch)

        resized_sunglasses = cv2.resize(
            sunglasses,
            None,
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_LINEAR,
        )

        # point_20 = landmarks.part(20)
        # point_25 = landmarks.part(25)
        # midpoint_y = (point_20.y + point_25.y) // 2
        # vertical_offset += midpoint_y - resized_sunglasses.shape[0] // 2

        sunglasses_width = resized_sunglasses.shape[1]
        sunglasses_height = resized_sunglasses.shape[0]
        sunglasses_3d_points = np.array(
            [
                [-sunglasses_width / 2, vertical_offset - sunglasses_height, 0],
                [sunglasses_width / 2, vertical_offset - sunglasses_height, 0],
                [-sunglasses_width / 2, vertical_offset, 0],
                [sunglasses_width / 2, vertical_offset, 0],
            ]
        )

        image_points_2d, _ = cv2.projectPoints(
            sunglasses_3d_points,
            corrected_rotation_vector,
            translation_vector,
            camera_matrix,
            dist_coeffs,
        )

        dst_points = image_points_2d.reshape(-1, 2)
        src_points = np.array(
            [
                [0, resized_sunglasses.shape[0]],
                [resized_sunglasses.shape[1], resized_sunglasses.shape[0]],
                [0, 0],
                [resized_sunglasses.shape[1], 0],
            ],
            dtype="float32",
        )

        transformation_matrix = self.compute_homography(src_points, dst_points)
        if transformation_matrix is not None:
            transformed_sunglasses = self.transform_overlay(
                resized_sunglasses, transformation_matrix, frame.shape
            )
            frame = self.alpha_blend(frame, transformed_sunglasses)

        return frame

    def apply_overlay(
        self,
        frame,
        overlay_img,
        camera_matrix,
        dist_coeffs,
        rotation_vector,
        translation_vector,
    ):
        """Apply a custom overlay to a frame based on facial landmarks"""
        angle = np.linalg.norm(-rotation_vector)

        offset_x = max(150, 150 + 100 * abs(angle))
        offset_y = 80

        model_points = np.array(
            [
                (0.0, 800.0, -50.0),  # Top of the forehead
                (0.0, -450.0, -50.0),  # Below the chin
                (-offset_x, offset_y, -50.0),  # Left side
                (offset_x, offset_y, -50.0),  # Right side
            ]
        )

        image_points, _ = cv2.projectPoints(
            model_points,
            -rotation_vector,
            translation_vector,
            camera_matrix,
            dist_coeffs,
        )
        image_points = image_points.reshape(-1, 2).astype("float32")

        overlay_points = np.array(
            [
                [overlay_img.shape[1] * 0.5, overlay_img.shape[0] * 0.05],
                [overlay_img.shape[1] * 0.5, overlay_img.shape[0] * 0.95],
                [overlay_img.shape[1] * 0.05, overlay_img.shape[0] * 0.5],
                [overlay_img.shape[1] * 0.95, overlay_img.shape[0] * 0.5],
            ],
            dtype="float32",
        )

        homography_matrix = self.compute_homography(overlay_points, image_points)
        if homography_matrix is not None:
            transformed_overlay = self.transform_overlay(
                overlay_img, homography_matrix, frame.shape
            )
            frame = self.alpha_blend(frame, transformed_overlay)

        return frame
