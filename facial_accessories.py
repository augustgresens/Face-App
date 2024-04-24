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


def add_mustache(frame, landmarks, mustache):
    src_points = np.array(
        [
            [0, 0],  # Top left corner
            [mustache.shape[1], 0],  # Top right corner
            [0, mustache.shape[0]],  # Bottom left corner
            [mustache.shape[1], mustache.shape[0]],  # Bottom right corner
        ],
        dtype="float32",
    )

    dst_points = np.array(
        [
            (landmarks.part(31).x, landmarks.part(31).y),  # Left point under the nose
            (landmarks.part(35).x, landmarks.part(35).y),  # Right point under the nose
            (landmarks.part(48).x, landmarks.part(48).y),  # Left mouth corner
            (landmarks.part(54).x, landmarks.part(54).y),  # Right mouth corner
        ],
        dtype="float32",
    )

    matrix, _ = cv2.findHomography(src_points, dst_points)
    if matrix is None:
        print("Homography could not be computed.")
        return frame

    transformed_mustache = cv2.warpPerspective(
        mustache, matrix, (frame.shape[1], frame.shape[0])
    )
    return alpha_blend(frame, transformed_mustache)


def add_sunglasses(frame, forehead_pts, left_eye_pts, right_eye_pts, sunglasses):
    if len(forehead_pts) < 9 or len(left_eye_pts) < 4 or len(right_eye_pts) < 3:
        return frame

    sunglasses_width = forehead_pts[8][0] - forehead_pts[0][0]
    sunglasses_height = max(
        left_eye_pts[3][1] - forehead_pts[2][1],
        right_eye_pts[2][1] - forehead_pts[2][1],
    )

    sunglasses_resized = cv2.resize(
        sunglasses, (int(sunglasses_width), int(sunglasses_height))
    )
    sunglasses_x = int(
        forehead_pts[0][0] + (sunglasses_width - sunglasses_resized.shape[1]) / 2
    )
    sunglasses_y = int(
        min(left_eye_pts[0][1], right_eye_pts[0][1], forehead_pts[0][1])
        + (sunglasses_height - sunglasses_resized.shape[0]) / 2
    )

    for c in range(3):
        frame[
            sunglasses_y : sunglasses_y + sunglasses_resized.shape[0],
            sunglasses_x : sunglasses_x + sunglasses_resized.shape[1],
            c,
        ] = sunglasses_resized[:, :, c] * (sunglasses_resized[:, :, 3] / 255.0) + frame[
            sunglasses_y : sunglasses_y + sunglasses_resized.shape[0],
            sunglasses_x : sunglasses_x + sunglasses_resized.shape[1],
            c,
        ] * (
            1.0 - sunglasses_resized[:, :, 3] / 255.0
        )

    return frame
