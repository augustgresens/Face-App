import tkinter as tk
from PIL import Image, ImageTk
import cv2
import dlib
import numpy as np
from facial_accessories import add_mustache, add_sunglasses
from pose_estimation import estimate_pose
from draw_axes import draw_axes

# Define global variables for landmarks and a flag to draw axes
forehead_pts = []
upper_lip_pts = []
left_eye_pts = []
right_eye_pts = []
nose_pts = []
mouth_pts = []
bottom_of_nose_y = 0
top_of_mouth_y = 0
draw_axes_flag = False
draw_sunglasses_flag = False
draw_mustache_flag = False


def show_frame():
    global forehead_pts, upper_lip_pts, left_eye_pts, right_eye_pts, nose_pts, mouth_pts, bottom_of_nose_y, top_of_mouth_y, draw_axes_flag
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        root.quit()
        return

    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(grayscale)

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
        frame, detector, predictor, camera_matrix, dist_coeffs
    )

    if success:
        if draw_axes_flag:
            frame = draw_axes(
                frame, rotation_vector, translation_vector, camera_matrix, dist_coeffs
            )

        # forehead landmarks (17 to 26)
        forehead_pts = [
            (landmarks.part(i).x, landmarks.part(i).y) for i in range(17, 27)
        ]

        # upper lip landmarks (48 to 59)
        upper_lip_pts = [
            (landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 60)
        ]

        # left eye landmarks (36 to 41)
        left_eye_pts = [
            (landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)
        ]

        # right eye landmarks (42 to 47)
        right_eye_pts = [
            (landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)
        ]

        # nose landmarks (27 to 34)
        nose_pts = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(27, 36)]

        # mouth landmarks (60 to 67)
        mouth_pts = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(60, 68)]

        # calculate the lowest y-value nose landmark
        bottom_of_nose_y = max(nose_pts[6][1], nose_pts[7][1])

        # calculate the highest y-value mouth landmark
        top_of_mouth_y = min(
            mouth_pts[1][1], mouth_pts[2][1], mouth_pts[3][1], mouth_pts[4][1]
        )

        # creates top-left point on the nose rectangle
        top_left_nose = (nose_pts[4][0], nose_pts[0][1])

        if draw_sunglasses_flag:
            frame = add_sunglasses(
                frame, forehead_pts, left_eye_pts, right_eye_pts, sunglasses
            )

        if draw_mustache_flag:
            frame = add_mustache(
                frame, upper_lip_pts, bottom_of_nose_y, top_of_mouth_y, mustache
            )

    # Convert to PIL Image
    cv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv_img)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("files/shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

mustache = cv2.imread("img/mustache.png", cv2.IMREAD_UNCHANGED)
sunglasses = cv2.imread("img/sunglasses.png", cv2.IMREAD_UNCHANGED)

# Setup GUI
root = tk.Tk()
root.bind("<Escape>", lambda e: root.quit())
lmain = tk.Label(root)
lmain.pack()


def toggle_draw_axes():
    global draw_axes_flag
    draw_axes_flag = not draw_axes_flag


def toggle_sunglasses():
    global draw_sunglasses_flag
    draw_sunglasses_flag = not draw_sunglasses_flag


def toggle_mustache():
    global draw_mustache_flag
    draw_mustache_flag = not draw_mustache_flag


def clear_decorations():
    global draw_axes_flag, draw_sunglasses_flag, draw_mustache_flag
    draw_axes_flag = draw_sunglasses_flag = draw_mustache_flag = False


# Setup GUI components (buttons for toggling axes drawing)
button_frame = tk.Frame(root)
button_frame.pack(side=tk.BOTTOM, fill=tk.X)

draw_axes_button = tk.Button(button_frame, text="Toggle Axes", command=toggle_draw_axes)
draw_axes_button.pack(side=tk.LEFT)

draw_sunglasses_button = tk.Button(
    button_frame, text="Toggle Sunglasses", command=toggle_sunglasses
)
draw_sunglasses_button.pack(side=tk.LEFT)

draw_mustache_button = tk.Button(
    button_frame, text="Toggle Mustache", command=toggle_mustache
)
draw_mustache_button.pack(side=tk.LEFT)

clear_axes_button = tk.Button(button_frame, text="Clear", command=clear_decorations)
clear_axes_button.pack(side=tk.LEFT)

# Start showing the frame
show_frame()
root.mainloop()

# Release the capture
cap.release()
cv2.destroyAllWindows()
