import numpy as np
from PIL import ImageGrab, Image
import imagehash
import time
import os

class MenuDetector:
    def __init__(self, menu_images_dir='menu_images', screenshot_dir='screenshots', hash_threshold=10):
        """
        Initializes the MenuDetector with a bounding box and hash comparison settings.

        :param bbox: Tuple defining the region of interest (x1, y1, x2, y2)
        :param menu_images_dir: Directory containing images of the menu for hash comparison
        :param screenshot_dir: Directory where screenshots will be saved
        :param hash_threshold: Threshold for hash difference to consider as a match
        """
        self.bbox = (2055, 150, 2555, 900)
        self.menu_images_dir = menu_images_dir
        self.screenshot_dir = screenshot_dir
        self.hash_threshold = hash_threshold
        self.screenshot_count = 0  # Counter for the screenshots

        # Load menu images and their hashes
        self.menu_hashes = self.load_menu_hashes()

        # Create the screenshots directory if it doesn't exist
        os.makedirs(self.screenshot_dir, exist_ok=True)

    def load_menu_hashes(self):
        """
        Loads images from the menu_images directory and computes their hashes.

        :return: List of image hashes.
        """
        menu_hashes = []
        for filename in os.listdir(self.menu_images_dir):
            if filename.endswith('.png') or filename.endswith('.jpg'):
                image_path = os.path.join(self.menu_images_dir, filename)
                image = Image.open(image_path)
                image_hash = imagehash.average_hash(image)
                menu_hashes.append(image_hash)
        return menu_hashes

    def capture_roi(self):
        """
        Captures the region of interest (ROI) from the screen.

        :return: PIL Image object of the captured ROI.
        """
        screenshot = ImageGrab.grab(bbox=self.bbox)
        return screenshot

    def save_screenshot(self, image):
        """
        Saves the captured image to the specified directory.

        :param image: PIL Image object to save.
        """
        screenshot_path = os.path.join(self.screenshot_dir, f'screenshot_{self.screenshot_count:04d}.png')
        image.save(screenshot_path)
        self.screenshot_count += 1

    def is_menu_open(self):
        """
        Checks if the menu is open by comparing the hash of the current ROI with known menu hashes.

        :return: Boolean value indicating if the menu is open.
        """
        roi_image = self.capture_roi()
        roi_hash = imagehash.average_hash(roi_image)

        # # Save the screenshot for inspection
        # self.save_screenshot(roi_image)

        # Compare the hash of the current ROI with known menu hashes
        for menu_hash in self.menu_hashes:
            hash_diff = roi_hash - menu_hash
            if hash_diff <= self.hash_threshold:
                return True

        return False