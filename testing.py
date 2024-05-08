import unittest
import cv2
import numpy as np
from unittest.mock import MagicMock
from facial_accessories import FacialAccessories


class TestFacialFilters(unittest.TestCase):

    def setUp(self):
        """Sets up by loading images and creating mock objects"""
        self.image = cv2.imread("img/jordan.png")
        self.mustache = cv2.imread("img/mustache.png", cv2.IMREAD_UNCHANGED)
        self.sunglasses = cv2.imread("img/sunglasses.png", cv2.IMREAD_UNCHANGED)
        assert (
            self.image is not None
            and self.mustache is not None
            and self.sunglasses is not None
        ), "Failed to load images"

        self.camera_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="double")
        self.dist_coeffs = np.zeros((4, 1), dtype="double")
        self.rotation_vector = np.zeros((3, 1), dtype="double")
        self.translation_vector = np.zeros((3, 1), dtype="double")

        self.facial_accessories = FacialAccessories()

    def test_with_null_image_input(self):
        """Test function behavior when given a null image as input"""
        result_frame_mustache = self.facial_accessories.add_mustache(
            None,
            self.mustache,
            self.camera_matrix,
            self.dist_coeffs,
            self.rotation_vector,
            self.translation_vector,
        )
        self.assertIsNone(
            result_frame_mustache,
            "Function should return None when add_mustache is given a null image.",
        )

    def test_no_faces_detected(self):
        """Test function behavior when no faces are detected in the image"""
        self.facial_accessories.compute_homography = MagicMock(return_value=None)
        result_image = self.facial_accessories.add_mustache(
            self.image.copy(),
            self.mustache,
            self.camera_matrix,
            self.dist_coeffs,
            self.rotation_vector,
            self.translation_vector,
        )
        self.assertTrue(
            np.array_equal(result_image, self.image),
            "Image should remain unchanged if no homography transformation is applied",
        )

    def test_add_mustache(self):
        """Test the add_mustache function"""
        result_image = self.facial_accessories.add_mustache(
            self.image.copy(),
            self.mustache,
            self.camera_matrix,
            self.dist_coeffs,
            self.rotation_vector,
            self.translation_vector,
        )
        self.assertIsNot(
            result_image, self.image, "Adding mustache should modify the image"
        )


if __name__ == "__main__":
    unittest.main()
