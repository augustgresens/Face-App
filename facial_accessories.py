import cv2
import numpy as np


def alpha_blend(frame, overlay, position=(0, 0)):
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


def add_mustache(
    frame,
    mustache,
    camera_matrix,
    dist_coeffs,
    rotation_vector,
    translation_vector,
):
    scale_factor = 0.5

    resized_mustache = cv2.resize(
        mustache, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR
    )

    vertical_offset = -50

    mustache_width = resized_mustache.shape[1]
    mustache_height = resized_mustache.shape[0]
    mustache_3d_points = np.array(
        [
            [-mustache_width / 2, vertical_offset, 0],  # Top-left corner
            [mustache_width / 2, vertical_offset, 0],  # Top-right corner
            [
                -mustache_width / 2,
                vertical_offset - mustache_height,
                0,
            ],  # Bottom-left corner
            [
                mustache_width / 2,
                vertical_offset - mustache_height,
                0,
            ],  # Bottom-right corner
        ]
    )

    image_points_2d, _ = cv2.projectPoints(
        mustache_3d_points,
        rotation_vector,
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

    transformation_matrix, _ = cv2.findHomography(src_points, dst_points)
    if transformation_matrix is None:
        print("Homography could not be computed.")
        return frame

    transformed_mustache = cv2.warpPerspective(
        resized_mustache, transformation_matrix, (frame.shape[1], frame.shape[0])
    )

    frame = alpha_blend(frame, transformed_mustache)
    return frame


def add_sunglasses(
    frame,
    landmarks,
    sunglasses,
    camera_matrix,
    dist_coeffs,
    rotation_vector,
    translation_vector,
):
    scale_factor = 1.7

    resized_sunglasses = cv2.resize(
        sunglasses,
        None,
        fx=scale_factor,
        fy=scale_factor,
        interpolation=cv2.INTER_LINEAR,
    )

    nose_bridge_y = landmarks.part(28).y
    vertical_offset = nose_bridge_y + int(resized_sunglasses.shape[0] / 1.9)

    sunglasses_width = resized_sunglasses.shape[1]
    sunglasses_height = resized_sunglasses.shape[0]
    sunglasses_3d_points = np.array(
        [
            [
                -sunglasses_width / 2,
                vertical_offset - sunglasses_height,
                0,
            ],  # Top-left corner
            [
                sunglasses_width / 2,
                vertical_offset - sunglasses_height,
                0,
            ],  # Top-right corner
            [-sunglasses_width / 2, vertical_offset, 0],  # Bottom-left corner
            [sunglasses_width / 2, vertical_offset, 0],  # Bottom-right corner
        ]
    )

    image_points_2d, _ = cv2.projectPoints(
        sunglasses_3d_points,
        rotation_vector,
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

    transformation_matrix, _ = cv2.findHomography(src_points, dst_points)
    if transformation_matrix is None:
        print("Homography could not be computed.")
        return frame

    transformed_sunglasses = cv2.warpPerspective(
        resized_sunglasses, transformation_matrix, (frame.shape[1], frame.shape[0])
    )

    frame = alpha_blend(frame, transformed_sunglasses)
    return frame


def apply_overlay(
    frame,
    landmarks,
    overlay_img,
    camera_matrix,
    dist_coeffs,
    rotation_vector,
    translation_vector,
):

    corrected_rotation_vector = -rotation_vector
    model_points = np.array(
        [
            (
                0.0,
                400.0,
                -100.0,
            ),  # Higher top of the forehead (adjusted outward and upward)
            (0.0, -300.0, -100.0),  # Below the chin
            (-300.0, 0.0, -100.0),  # Left side of the face near the ear
            (300.0, 0.0, -100.0),  # Right side of the face near the ear
        ]
    )

    image_points, _ = cv2.projectPoints(
        model_points,
        corrected_rotation_vector,
        translation_vector,
        camera_matrix,
        dist_coeffs,
    )
    image_points = image_points.reshape(-1, 2).astype("float32")

    overlay_points = np.array(
        [
            [
                overlay_img.shape[1] * 0.5,
                overlay_img.shape[0] * 0.05,
            ],  # Top middle for the forehead
            [
                overlay_img.shape[1] * 0.5,
                overlay_img.shape[0] * 0.9,
            ],  # Bottom middle for below the chin
            [
                overlay_img.shape[1] * 0.05,
                overlay_img.shape[0] * 0.5,
            ],  # Left middle height
            [
                overlay_img.shape[1] * 0.95,
                overlay_img.shape[0] * 0.5,
            ],  # Right middle height
        ],
        dtype="float32",
    )

    homography_matrix, _ = cv2.findHomography(overlay_points, image_points)

    transformed_overlay = cv2.warpPerspective(
        overlay_img, homography_matrix, (frame.shape[1], frame.shape[0])
    )

    return alpha_blend(frame, transformed_overlay)
