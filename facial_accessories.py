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


def add_mustache(frame, upper_lip_pts, bottom_of_nose_y, top_of_mouth_y, mustache):
    if frame is None:
        return None

    if len(upper_lip_pts) < 7:
        return frame

    mustache_width = upper_lip_pts[6][0] - upper_lip_pts[0][0]
    mustache_height = top_of_mouth_y - bottom_of_nose_y

    mustache_resized = cv2.resize(mustache, (int(mustache_width), int(mustache_height)))

    mustache_x = int(
        upper_lip_pts[0][0] + (mustache_width - mustache_resized.shape[1]) / 2
    )
    mustache_y = int(
        bottom_of_nose_y + (mustache_height - mustache_resized.shape[0]) / 2
    )

    for c in range(3):
        frame[
            mustache_y : mustache_y + mustache_resized.shape[0],
            mustache_x : mustache_x + mustache_resized.shape[1],
            c,
        ] = mustache_resized[:, :, c] * (mustache_resized[:, :, 3] / 255.0) + frame[
            mustache_y : mustache_y + mustache_resized.shape[0],
            mustache_x : mustache_x + mustache_resized.shape[1],
            c,
        ] * (
            1.0 - mustache_resized[:, :, 3] / 255.0
        )

    return frame


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
