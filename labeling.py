import os
import tkinter as tk
from PIL import Image, ImageTk
import json

class ImageLabeler:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.image_count = 2932  # Adjust if you have more or fewer pairs
        self.labels = []
        self.load_existing_labels()  # Load existing labels if any

        # Set up the GUI
        self.root = tk.Tk()
        self.root.title("Movement Detection Labeling Tool")

        # Image displays and labels for filenames
        self.filename1_label = tk.Label(self.root, text="", font=('Helvetica', 12))
        self.filename1_label.pack()
        self.img_label1 = tk.Label(self.root)
        self.img_label1.pack(side="left")

        self.filename2_label = tk.Label(self.root, text="", font=('Helvetica', 12))
        self.filename2_label.pack()
        self.img_label2 = tk.Label(self.root)
        self.img_label2.pack(side="left")

        # Buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(fill=tk.X, expand=True)

        same_btn = tk.Button(button_frame, text="No Movement", command=lambda: self.save_label(0))
        same_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)

        diff_btn = tk.Button(button_frame, text="Movement Detected", command=lambda: self.save_label(1))
        diff_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)

        ledge_btn = tk.Button(button_frame, text="Ledge", command=lambda: self.save_label(2))
        ledge_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)

        # Load the first image pair
        self.load_images()

        self.root.mainloop()

    def load_existing_labels(self):
        try:
            with open('label_data.json', 'r') as f:
                existing_data = json.load(f)
                self.labels = [item['label'] for item in existing_data]
                self.index = len(self.labels)  # Start at the next index after last labeled pair
        except FileNotFoundError:
            self.index = 0  # Start from beginning if no existing data

    def load_images(self):
        if self.index < self.image_count:
            img1_filename = f'image1_{self.index:04d}.png'
            img2_filename = f'image2_{self.index:04d}.png'
            img1_path = os.path.join(self.folder_path, img1_filename)
            img2_path = os.path.join(self.folder_path, img2_filename)

            img1 = Image.open(img1_path)
            img2 = Image.open(img2_path)

            self.photo_img1 = ImageTk.PhotoImage(img1)
            self.photo_img2 = ImageTk.PhotoImage(img2)

            self.img_label1.configure(image=self.photo_img1)
            self.img_label2.configure(image=self.photo_img2)

            self.filename1_label.config(text=img1_filename)
            self.filename2_label.config(text=img2_filename)

            self.index += 1  # Move to the next pair

    def save_label(self, label):
        self.labels.append(label)
        self.save_intermediate_results()  # Save after each labeling to preserve progress
        if self.index >= self.image_count:
            self.finish_labeling()
        else:
            self.load_images()

    def save_intermediate_results(self):
        results = [{'image1': f'image1_{i:04d}.png', 'image2': f'image2_{i:04d}.png', 'label': label} for i, label in enumerate(self.labels)]
        with open('label_data.json', 'w') as f:
            json.dump(results, f, indent=4)
        print("Intermediate results saved.")

    def finish_labeling(self):
        print("Labeling complete. Results saved to label_data.json.")
        self.root.destroy()

# Example usage: create an object with the path to your images
if __name__ == "__main__":
    path_to_images = "screenshots"
    app = ImageLabeler(path_to_images)
