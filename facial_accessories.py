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
    overlay = overlay[0 : (y_end - y), 0 : (x_end - x)]

    alpha_overlay = overlay[:, :, 3] / 255.0
    alpha_frame = 1.0 - alpha_overlay
    for c in range(3):
        roi[:, :, c] = alpha_overlay * overlay[:, :, c] + alpha_frame * roi[:, :, c]

    frame[y:y_end, x:x_end] = roi
    return frame


def apply_overlay(frame, landmarks, overlay_img, overlay_points, landmark_indices):
    image_points_2d = np.array(
        [(landmarks.part(i).x, landmarks.part(i).y) for i in landmark_indices],
        dtype="float32",
    )

    matrix, _ = cv2.findHomography(overlay_points, image_points_2d)
    transformed_overlay = cv2.warpPerspective(
        overlay_img, matrix, (frame.shape[1], frame.shape[0])
    )

    return alpha_blend(frame, transformed_overlay)


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
    # Using eye corners for horizontal scaling and nose bridge for vertical positioning
    eye_distance = np.linalg.norm(
        [
            (landmarks.part(45).x - landmarks.part(36).x),
            (landmarks.part(45).y - landmarks.part(36).y),
        ]
    )
    scale_factor = 1.5

    resized_sunglasses = cv2.resize(
        sunglasses,
        None,
        fx=scale_factor,
        fy=scale_factor,
        interpolation=cv2.INTER_LINEAR,
    )

    # Positioning the sunglasses at the nose bridge
    # Adjust the 'y' position to move the sunglasses up or down relative to the nose bridge
    nose_bridge_y = landmarks.part(28).y
    vertical_offset = nose_bridge_y + int(resized_sunglasses.shape[0])

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
