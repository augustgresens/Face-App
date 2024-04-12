import tkinter as tk
from PIL import Image, ImageTk
import cv2
import dlib
import numpy as np
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
        self.sunglasses = cv2.imread("img/sunglasses.png", cv2.IMREAD_UNCHANGED)

        self.forehead_pts = []
        self.upper_lip_pts = []
        self.left_eye_pts = []
        self.right_eye_pts = []
        self.nose_pts = []
        self.mouth_pts = []
        self.bottom_of_nose_y = 0
        self.top_of_mouth_y = 0

        self.draw_axes_flag = False
        self.draw_sunglasses_flag = False
        self.draw_mustache_flag = False

        self.root = tk.Tk()
        self.root.bind("<Escape>", lambda e: self.root.quit())
        self.lmain = tk.Label(self.root)
        self.lmain.pack()

        self.setup_ui()
        self.show_frame()
        self.root.mainloop()

    def toggle_draw_axes(self):
        self.draw_axes_flag = not self.draw_axes_flag

    def toggle_sunglasses(self):
        self.draw_sunglasses_flag = not self.draw_sunglasses_flag

    def toggle_mustache(self):
        self.draw_mustache_flag = not self.draw_mustache_flag

    def clear_decorations(self):
        self.draw_axes_flag = self.draw_sunglasses_flag = self.draw_mustache_flag = (
            False
        )

    def setup_ui(self):
        button_frame = tk.Frame(self.root)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X)

        draw_axes_button = tk.Button(
            button_frame, text="Toggle Axes", command=self.toggle_draw_axes
        )
        draw_axes_button.pack(side=tk.LEFT)

        draw_sunglasses_button = tk.Button(
            button_frame, text="Toggle Sunglasses", command=self.toggle_sunglasses
        )
        draw_sunglasses_button.pack(side=tk.LEFT)

        draw_mustache_button = tk.Button(
            button_frame, text="Toggle Mustache", command=self.toggle_mustache
        )
        draw_mustache_button.pack(side=tk.LEFT)

        clear_axes_button = tk.Button(
            button_frame, text="Clear", command=self.clear_decorations
        )
        clear_axes_button.pack(side=tk.LEFT)

    def show_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to capture frame")
            self.root.quit()
            return

        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(grayscale)

        # Camera internals
        size = frame.shape
        focal_length = size[1]
        center = (size[1] // 2, size[0] // 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype="double",
        )
        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

        success, rotation_vector, translation_vector, landmarks = estimate_pose(
            frame, self.detector, self.predictor, camera_matrix, dist_coeffs
        )

        if success:
            if self.draw_axes_flag:
                frame = draw_axes(
                    frame,
                    rotation_vector,
                    translation_vector,
                    camera_matrix,
                    dist_coeffs,
                )

            if landmarks:
                # Define additional processing here for decorations based on landmarks
                self.process_landmarks(landmarks)

            if self.draw_sunglasses_flag:
                frame = add_sunglasses(
                    frame,
                    self.forehead_pts,
                    self.left_eye_pts,
                    self.right_eye_pts,
                    self.sunglasses,
                )

            if self.draw_mustache_flag:
                frame = add_mustache(
                    frame,
                    self.upper_lip_pts,
                    self.bottom_of_nose_y,
                    self.top_of_mouth_y,
                    self.mustache,
                )

        # Convert to PIL Image
        cv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv_img)
        imgtk = ImageTk.PhotoImage(image=img)
        self.lmain.imgtk = imgtk
        self.lmain.configure(image=imgtk)
        self.lmain.after(10, self.show_frame)

    def process_landmarks(self, landmarks):
        # Update landmark-based measurements and points for decorations
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
