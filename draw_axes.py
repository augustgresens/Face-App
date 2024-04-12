import cv2
import numpy as np


def draw_axes(frame, rotation_vector, translation_vector, camera_matrix, dist_coeffs):
    axis_length = 200
    axis_3d = np.float32(
        [[axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length], [0, 0, 0]]
    ).reshape(-1, 3)

    axis_2d, _ = cv2.projectPoints(
        axis_3d, rotation_vector, translation_vector, camera_matrix, dist_coeffs
    )
    axis_2d = axis_2d.reshape(-1, 2)

    frame = cv2.line(
        frame,
        tuple(axis_2d[3].astype(int)),
        tuple(axis_2d[0].astype(int)),
        (0, 0, 255),
        1,
    )
    frame = cv2.line(
        frame,
        tuple(axis_2d[3].astype(int)),
        tuple(axis_2d[1].astype(int)),
        (0, 255, 0),
        1,
    )
    frame = cv2.line(
        frame,
        tuple(axis_2d[3].astype(int)),
        tuple(axis_2d[2].astype(int)),
        (255, 0, 0),
        1,
    )

    return frame
