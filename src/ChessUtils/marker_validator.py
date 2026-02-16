import tkinter as tk
from PIL import Image, ImageTk
import cv2


class MarkerValidator:
    def __init__(self, img, callback_ok=None, callback_retry=None):
        self.callback_ok = callback_ok
        self.callback_retry = callback_retry

        # Image OpenCV → PIL (stockée en original)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.pil_img_original = Image.fromarray(img_rgb)

        self.root = tk.Tk()
        self.root.title("Validation des marqueurs")
        self.root.geometry("900x700")

        # --- Layout ---
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        # Zone image
        self.image_label = tk.Label(self.root, bg="black")
        self.image_label.grid(row=0, column=0, sticky="nsew")

        # Zone boutons
        button_frame = tk.Frame(self.root)
        button_frame.grid(row=1, column=0, pady=10)

        tk.Button(button_frame, text="OK", width=12, command=self.on_ok).pack(
            side=tk.LEFT, padx=10
        )
        tk.Button(button_frame, text="Retry", width=12, command=self.on_retry).pack(
            side=tk.LEFT, padx=10
        )

        self.result = None
        self.tk_img = None

        # Bind resize
        self.root.bind("<Configure>", self.on_resize)

        self.root.mainloop()

    def on_resize(self, event):
        """Redimensionne l'image pour s'adapter à la fenêtre."""
        if event.widget != self.root:
            return

        # Taille disponible pour l'image
        frame_width = self.root.winfo_width()
        frame_height = self.root.winfo_height() - 80  # place pour boutons

        if frame_width <= 0 or frame_height <= 0:
            return

        # Conserver le ratio
        img_w, img_h = self.pil_img_original.size
        scale = min(frame_width / img_w, frame_height / img_h)

        new_size = (int(img_w * scale), int(img_h * scale))
        resized = self.pil_img_original.resize(new_size, Image.LANCZOS)

        self.tk_img = ImageTk.PhotoImage(resized)
        self.image_label.config(image=self.tk_img)

    def on_ok(self):
        self.result = True
        if self.callback_ok:
            self.callback_ok()
        self.root.destroy()

    def on_retry(self):
        self.result = False
        if self.callback_retry:
            self.callback_retry()
        self.root.destroy()