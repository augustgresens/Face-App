import cv2
import numpy as np
import tkinter as tk
import dlib
from gui import GUI
from facial_accessories import add_mustache, add_sunglasses
from pose_estimation import estimate_pose
from draw_axes import draw_axes


class FaceApp:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(
            "files/shape_predictor_68_face_landmarks.dat"
        )
        self.cap = cv2.VideoCapture(0)

        self.mustache = cv2.imread("img/mustache.png", cv2.IMREAD_UNCHANGED)
        if self.mustache is None:
            raise FileNotFoundError("Mustache image not found.")

        self.sunglasses = cv2.imread("img/sunglasses.png", cv2.IMREAD_UNCHANGED)
        if self.sunglasses is None:
            raise FileNotFoundError("Sunglasses image not found.")

        self.flags = {"axes": False, "sunglasses": False, "mustache": False}

        self.root = tk.Tk()
        self.root.bind("<Escape>", lambda e: self.root.quit())
        self.gui = GUI(self.root, self.update_flags)

        self.show_frame()
        self.root.mainloop()

    def setup_camera(self):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("files/shape_predictor_68_face_landmarks.dat")
        cap = cv2.VideoCapture(0)
        return detector, predictor, cap

    def update_flags(self, flag):
        if flag == "clear":
            for key in self.flags:
                self.flags[key] = False
        else:
            self.flags[flag] = not self.flags[flag]

    def process_frame(self, frame):
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(grayscale)

        size = frame.shape
        focal_length = size[1]
        center = (size[1] // 2, size[0] // 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype="double",
        )
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, translation_vector, landmarks = estimate_pose(
            frame, self.detector, self.predictor, camera_matrix, dist_coeffs
        )

        if success:
            if self.flags["axes"]:
                frame = draw_axes(
                    frame,
                    rotation_vector,
                    translation_vector,
                    camera_matrix,
                    dist_coeffs,
                )

            if landmarks:
                self.process_landmarks(landmarks)

            if self.flags["sunglasses"]:
                frame = add_sunglasses(
                    frame,
                    self.forehead_pts,
                    self.left_eye_pts,
                    self.right_eye_pts,
                    self.sunglasses,
                )

            if self.flags["mustache"]:
                frame = add_mustache(
                    frame,
                    self.upper_lip_pts,
                    self.bottom_of_nose_y,
                    self.top_of_mouth_y,
                    self.mustache,
                )

        return frame

    def show_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to capture frame")
            self.root.quit()
            return

        frame = self.process_frame(frame)
        self.gui.update_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.root.after(10, self.show_frame)

    def process_landmarks(self, landmarks):
        self.forehead_pts = [
            (landmarks.part(i).x, landmarks.part(i).y) for i in range(17, 27)
        ]
        self.upper_lip_pts = [
            (landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 60)
        ]
        self.left_eye_pts = [
            (landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)
        ]
        self.right_eye_pts = [
            (landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)
        ]
        self.nose_pts = [
            (landmarks.part(i).x, landmarks.part(i).y) for i in range(27, 36)
        ]
        self.mouth_pts = [
            (landmarks.part(i).x, landmarks.part(i).y) for i in range(60, 68)
        ]
        self.bottom_of_nose_y = max(self.nose_pts[6][1], self.nose_pts[7][1])
        self.top_of_mouth_y = min(
            self.mouth_pts[1][1],
            self.mouth_pts[2][1],
            self.mouth_pts[3][1],
            self.mouth_pts[4][1],
        )


if __name__ == "__main__":
    app = FaceApp()
