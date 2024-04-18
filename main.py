import cv2
import tkinter as tk
import dlib
from gui import GUI
from frame_processor import FrameProcessor


class FaceApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.bind("<Escape>", lambda e: self.root.quit())

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(
            "data_files/shape_predictor_68_face_landmarks.dat"
        )

        self.sunglasses = cv2.imread("img/sunglasses.png", cv2.IMREAD_UNCHANGED)
        if self.sunglasses is None:
            raise FileNotFoundError("Sunglasses image not found.")

        self.mustache = cv2.imread("img/mustache.png", cv2.IMREAD_UNCHANGED)
        if self.mustache is None:
            raise FileNotFoundError("Mustache image not found.")

        self.overlay_img = cv2.imread("img/peacockbass.png", cv2.IMREAD_UNCHANGED)
        if self.overlay_img is None:
            raise FileNotFoundError("Overlay image not found.")

        self.processor = FrameProcessor(
            self.detector,
            self.predictor,
            self.sunglasses,
            self.mustache,
            self.overlay_img,
        )

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Camera could not be opened.")

        self.gui = GUI(self.root, self.update_flags)
        self.flags = {
            "axes": False,
            "sunglasses": False,
            "mustache": False,
            "overlay": False,
        }

        self.show_frame()
        self.root.mainloop()

    def update_flags(self, flag):
        if flag == "clear":
            for key in self.flags:
                self.flags[key] = False
        else:
            self.flags[flag] = not self.flags[flag]

    def show_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to capture frame")
            self.cap.release()
            self.root.quit()
            return

        frame = self.processor.process_frame(frame, self.flags)
        self.gui.update_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.root.after(10, self.show_frame)


if __name__ == "__main__":
    app = FaceApp()
