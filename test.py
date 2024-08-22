import cv2
import numpy as np
from PIL import ImageGrab, Image
import time

def capture_screenshot(bbox, resize_factor=0.25):
    screenshot = ImageGrab.grab(bbox=bbox)
    screenshot = screenshot.resize(
        (int(screenshot.width * resize_factor), int(screenshot.height * resize_factor)), Image.BILINEAR
    )
    grayscale_screenshot = screenshot.convert('L')
    return np.array(grayscale_screenshot)

def detect_movement_sift(image1, image2, min_match=10):
    sift = cv2.SIFT_create()
    
    # Detect keypoints and compute descriptors
    kps1, des1 = sift.detectAndCompute(image1, None)
    kps2, des2 = sift.detectAndCompute(image2, None)
    
    # Check if keypoints and descriptors were found in both images
    if des1 is None or des2 is None:
        return 0, 0, False  # No keypoints found, so no movement detected
    
    # FLANN based matcher
    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Store all the good matches as per Lowe's ratio test.
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    if len(good_matches) > min_match:
        src_pts = np.float32([kps1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kps2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()

        h, w = image1.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        # Calculate the shift direction based on the transformation matrix M
        shift_x = M[0, 2]
        shift_y = M[1, 2]
        return shift_x, shift_y, True  # Movement detected

    else:
        return 0, 0, False  # No movement detected

def detect_movement():
    bbox = (1250, 110, 2600, 1100)
    prev_image = capture_screenshot(bbox)
    
    while True:
        time.sleep(1)
        current_image = capture_screenshot(bbox)
        
        shift_x, shift_y, movement_detected = detect_movement_sift(prev_image, current_image)

        if abs(shift_x) > 5 or abs(shift_y) > 5:
            if abs(shift_x) > abs(shift_y):
                if shift_x > 5:
                    print(f"Detected movement to the left: x_shift={shift_x}")
                elif shift_x < -5:
                    print(f"Detected movement to the right: x_shift={shift_x}")
            else:
                if shift_y > 5:
                    print(f"Detected movement up: y_shift={shift_y}")
                elif shift_y < -5:
                    print(f"Detected movement down: y_shift={shift_y}")
        else:
            print("No significant movement detected.")


        prev_image = current_image

# Example usage
time.sleep(1)
detect_movement()
