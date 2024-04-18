import unittest
import cv2
import numpy as np
from unittest.mock import MagicMock
from facial_accessories import add_mustache, add_sunglasses
from pose_estimation import estimate_pose


class TestFacialFilters(unittest.TestCase):

    def setUp(self):
        self.image = cv2.imread("img/face.jpg")
        self.mustache = cv2.imread("img/mustache.png", cv2.IMREAD_UNCHANGED)
        self.sunglasses = cv2.imread("img/sunglasses.png", cv2.IMREAD_UNCHANGED)
        assert (
            self.image is not None
            and self.mustache is not None
            and self.sunglasses is not None
        ), "Failed to load images"

        self.mock_detector = MagicMock(return_value=[])
        self.mock_predictor = MagicMock()

    def test_with_null_image_input(self):
        result_frame_mustache = add_mustache(None, [], 0, 0, self.mustache)
        result_frame_sunglasses = add_sunglasses(None, [], [], [], self.sunglasses)

        self.assertIsNone(
            result_frame_mustache,
            "Function should return None or an error when add_mustache is given a null image.",
        )
        self.assertIsNone(
            result_frame_sunglasses,
            "Function should return None or an error when add_sunglasses is given a null image.",
        )

    def test_no_faces_detected(self):
        camera_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="double")
        dist_coeffs = np.zeros((4, 1), dtype="double")
        result = estimate_pose(
            self.image,
            self.mock_detector,
            self.mock_predictor,
            camera_matrix,
            dist_coeffs,
        )
        self.assertEqual(
            result,
            (None, None, None, None),
            "Function should return None for all outputs when no faces are detected",
        )

    def test_add_mustache(self):
        upper_lip_pts = [
            (10, 10),
            (20, 20),
            (30, 10),
            (40, 20),
            (50, 10),
            (60, 20),
            (70, 10),
        ]
        bottom_of_nose_y = 5
        top_of_mouth_y = 15
        result_image = add_mustache(
            self.image.copy(),
            upper_lip_pts,
            bottom_of_nose_y,
            top_of_mouth_y,
            self.mustache,
        )
        difference = np.any(self.image != result_image)
        self.assertTrue(difference, "Adding mustache should have modified the frame")


if __name__ == "__main__":
    unittest.main()
