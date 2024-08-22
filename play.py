from PIL import ImageGrab
import time
import os
import hashlib
import imagehash

def capture_region_screenshot(bbox, interval=1, output_folder='screenshots', hash_size=8):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    count = 0
    image_hashes = {}  # Dictionary to store image hashes

    while True:
        # Capture the specified region
        screenshot = ImageGrab.grab(bbox=bbox)

        # Compute the perceptual hash of the screenshot
        perceptual_hash = imagehash.average_hash(screenshot, hash_size=hash_size)

        # Convert the perceptual hash to a string for easier comparison
        screenshot_hash = str(perceptual_hash)

        # Check if a similar hash is already in the dictionary
        if not any(perceptual_hash - existing_hash < 5 for existing_hash in image_hashes.values()):
            # Save the screenshot only if it's new or significantly different
            filename = os.path.join(output_folder, f'screenshot_{count:04d}.png')
            screenshot.save(filename)
            image_hashes[screenshot_hash] = perceptual_hash  # Store the hash in the dictionary
            
            print(f"New screenshot saved: {filename}")
            count += 1
        else:
            print("Screenshot is visually similar to an existing one.")

        # Wait for the specified interval
        time.sleep(interval)

# Example: Manually set the coordinates for the region you want to capture
# Adjust these coordinates to match your DeSmuME window
left = 1250  # Adjust left coordinate
top = 110    # Adjust top coordinate
right = 2600 # Adjust right coordinate
bottom = 1100 # Adjust bottom coordinate

capture_region_screenshot(bbox=(left, top, right, bottom), interval=1)
