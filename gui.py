import tkinter as tk
from PIL import Image, ImageTk


class GUI:
    def __init__(self, parent, update_callback):
        self.parent = parent
        self.update_callback = update_callback
        self.setup_gui()

    def setup_gui(self):
        self.lmain = tk.Label(self.parent)
        self.lmain.pack()

        button_frame = tk.Frame(self.parent)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X)

        tk.Button(
            button_frame,
            text="Toggle Axes",
            command=lambda: self.update_callback("axes"),
        ).pack(side=tk.LEFT)
        tk.Button(
            button_frame,
            text="Toggle Sunglasses",
            command=lambda: self.update_callback("sunglasses"),
        ).pack(side=tk.LEFT)
        tk.Button(
            button_frame,
            text="Toggle Mustache",
            command=lambda: self.update_callback("mustache"),
        ).pack(side=tk.LEFT)
        tk.Button(
            button_frame,
            text="Toggle Overlay",
            command=lambda: self.update_callback("overlay"),
        ).pack(side=tk.LEFT)
        tk.Button(
            button_frame, text="Clear", command=lambda: self.update_callback("clear")
        ).pack(side=tk.LEFT)

    def update_image(self, image):
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(image))
        self.lmain.imgtk = imgtk
        self.lmain.configure(image=imgtk)

    def mainloop(self):
        self.parent.mainloop()
