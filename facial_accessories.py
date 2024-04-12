import numpy as np
import cv2
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("files/shape_predictor_68_face_landmarks.dat")


def add_mustache(frame, upper_lip_pts, bottom_of_nose_y, top_of_mouth_y, mustache):
    # Check if there are enough points in upper_lip_pts
    if len(upper_lip_pts) < 7:
        return frame  # Return the frame unmodified if there are not enough points

    # Existing functionality to calculate mustache dimensions and placement
    mustache_width = upper_lip_pts[6][0] - upper_lip_pts[0][0]
    mustache_height = top_of_mouth_y - bottom_of_nose_y

    mustache_resized = cv2.resize(mustache, (int(mustache_width), int(mustache_height)))

    mustache_x = int(
        upper_lip_pts[0][0] + (mustache_width - mustache_resized.shape[1]) / 2
    )
    mustache_y = int(
        bottom_of_nose_y + (mustache_height - mustache_resized.shape[0]) / 2
    )

    # Adds the mustache to the frame
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
    # Ensure that there are enough points in the forehead_pts to calculate width and in eye_pts for height
    if len(forehead_pts) < 9 or len(left_eye_pts) < 4 or len(right_eye_pts) < 3:
        return frame  # Or raise an exception if that's preferred

    # Existing functionality for calculating sunglasses dimensions and placement
    sunglasses_width = forehead_pts[8][0] - forehead_pts[0][0]
    sunglasses_height = max(
        left_eye_pts[3][1] - forehead_pts[2][1],  # Ensure this index is available
        right_eye_pts[2][1] - forehead_pts[2][1],  # Adjusted index if necessary
    )

    # Continue with resizing and placing sunglasses
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

    # Overlay sunglasses onto frame
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
