import cv2
import numpy as np


def estimate_pose(frame, detector, predictor, camera_matrix, dist_coeffs):
    # Ensure the frame is converted to grayscale only if it is not already
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        grayscale = frame  # The frame is already in grayscale

    faces = detector(grayscale)
    for face in faces:
        landmarks = predictor(grayscale, face)
        model_points = np.array(
            [
                (0.0, 0.0, 0.0),  # Nose tip
                (0.0, -330.0, -65.0),  # Chin
                (-225.0, 170.0, -135.0),  # Left eye left corner
                (225.0, 170.0, -135.0),  # Right eye right corner
                (-150.0, -150.0, -125.0),  # Left Mouth corner
                (150.0, -150.0, -125.0),  # Right mouth corner
                (0.0, -150.0, -125.0),  # Nose Bridge
            ],
            dtype="double",
        )

        image_points = np.array(
            [
                (landmarks.part(30).x, landmarks.part(30).y),  # Nose tip
                (landmarks.part(8).x, landmarks.part(8).y),  # Chin
                (landmarks.part(36).x, landmarks.part(36).y),  # Left eye left corner
                (landmarks.part(45).x, landmarks.part(45).y),  # Right eye right corner
                (landmarks.part(48).x, landmarks.part(48).y),  # Left Mouth corner
                (landmarks.part(54).x, landmarks.part(54).y),  # Right mouth corner
                (landmarks.part(28).x, landmarks.part(28).y),  # Nose bridge
            ],
            dtype="double",
        )

        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        return success, rotation_vector, translation_vector, landmarks

    # If no faces are detected, return None for all components
    return None, None, None, None
