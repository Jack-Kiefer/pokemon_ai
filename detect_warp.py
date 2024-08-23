import numpy as np
from PIL import ImageGrab, Image
import time

def capture_screenshot(bbox=None, resize_factor=1.0):
    # Capture a screenshot of the specified bounding box
    screenshot = ImageGrab.grab(bbox=bbox)
    
    # Optionally resize the screenshot
    if resize_factor != 1.0:
        screenshot = screenshot.resize(
            (int(screenshot.width * resize_factor), int(screenshot.height * resize_factor)), 
            Image.BILINEAR
        )
    
    # Convert the screenshot to grayscale
    grayscale_screenshot = screenshot.convert('L')
    
    # Convert to a NumPy array for processing
    return np.array(grayscale_screenshot)

def is_black_screen(image, threshold=20):
    # Check if the image is mostly black
    # The threshold value indicates the maximum pixel value to be considered "black"
    return np.mean(image) < threshold

def detect_warp():
    bbox = (1270, 115, 2580, 1100)  # Adjust the bounding box as needed
    
    while True:
        time.sleep(1)  # Take a screenshot every second
        
        # Capture the screenshot
        screenshot = capture_screenshot(bbox=bbox)
        
        # Check if the screen is black
        if is_black_screen(screenshot):
            print("Warp detected: screen is black.")
        # No need to print anything if the screen is not black

# Example usage
time.sleep(1)  # Give some time before starting
detect_warp()
