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

        draw_axes_button = tk.Button(
            button_frame,
            text="Toggle Axes",
            command=lambda: self.update_callback("axes"),
        )
        draw_axes_button.pack(side=tk.LEFT)

        draw_sunglasses_button = tk.Button(
            button_frame,
            text="Toggle Sunglasses",
            command=lambda: self.update_callback("sunglasses"),
        )
        draw_sunglasses_button.pack(side=tk.LEFT)

        draw_mustache_button = tk.Button(
            button_frame,
            text="Toggle Mustache",
            command=lambda: self.update_callback("mustache"),
        )
        draw_mustache_button.pack(side=tk.LEFT)

        clear_axes_button = tk.Button(
            button_frame, text="Clear", command=lambda: self.update_callback("clear")
        )
        clear_axes_button.pack(side=tk.LEFT)

    def update_image(self, image):
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(image))
        self.lmain.imgtk = imgtk
        self.lmain.configure(image=imgtk)

    def mainloop(self):
        self.parent.mainloop()
