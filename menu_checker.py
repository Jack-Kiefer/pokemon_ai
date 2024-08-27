import numpy as np
from PIL import ImageGrab, Image
import imagehash
import time
import os

class MenuDetector:
    def __init__(self, menu_images_dir='menu_images', screenshot_dir='menu_shots', hash_threshold=2):
        """
        Initializes the MenuDetector with settings for menu detection.
        
        :param menu_images_dir: Directory containing images of the menu for hash comparison
        :param screenshot_dir: Directory where screenshots will be saved
        :param hash_threshold: Threshold for hash difference to consider as a match
        """
        self.bbox = (2055, 150, 2555, 900)
        self.fullbbox = (1275, 135, 2565, 1100)  # Coordinates for the region you want to capture
        self.menu_images_dir = menu_images_dir
        self.screenshot_dir = screenshot_dir
        self.hash_threshold = hash_threshold
        self.screenshot_count = 0

        # Load menu images and their hashes
        self.menu_hashes = self.load_menu_hashes()
        
        # Create the screenshots directory if it doesn't exist
        os.makedirs(self.screenshot_dir, exist_ok=True)

    def load_menu_hashes(self):
        """
        Loads images from the menu_images directory and computes their hashes.
        """
        menu_hashes = []
        for filename in os.listdir(self.menu_images_dir):
            if filename.endswith('.png') or filename.endswith('.jpg'):
                image_path = os.path.join(self.menu_images_dir, filename)
                image = Image.open(image_path)
                image_hash = imagehash.average_hash(image)
                menu_hashes.append(image_hash)
        return menu_hashes

    def capture_roi(self, bbox):
        """
        Captures the region of interest (ROI) from the screen.
        """
        screenshot = ImageGrab.grab(bbox=bbox)
        return screenshot

    def is_screen_mostly_white(self, image, threshold=10):
        """
        Determines if the captured screen is mostly white.

        :param image: PIL Image object of the captured screen.
        :return: Boolean indicating if the screen is mostly white.
        """
        return np.mean(image) < threshold or np.mean(image) > 255 - threshold

    def is_menu_open(self):
        """
        Checks if the menu is open by comparing the hash of the current ROI with known menu hashes.
        """
        roi_image = self.capture_roi(self.bbox)

        # Check for mostly white screen to avoid false positives
        if self.is_screen_mostly_white(self.capture_roi(self.fullbbox)):
            return False
        
        roi_hash = imagehash.average_hash(roi_image)
        for menu_hash in self.menu_hashes:
            hash_diff = roi_hash - menu_hash
            if hash_diff <= self.hash_threshold:
                # print("menu open")
                # self.save_screenshot()  # Call the method to save the screenshot
                return True
        return False

    def save_screenshot(self):
        """
        Saves a screenshot of the full region to the screenshots directory.
        """
        screenshot = self.capture_roi(self.fullbbox)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        screenshot_path = os.path.join(self.screenshot_dir, f"screenshot_{timestamp}_{self.screenshot_count}.png")
        screenshot.save(screenshot_path)
        self.screenshot_count += 1
        print(f"Screenshot saved to {screenshot_path}")
