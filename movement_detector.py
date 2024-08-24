# movement_detector.py

import numpy as np
import cv2
from PIL import Image

def save_screenshot(image, name, screenshot_dir='screenshots', count=0):
    """
    Save a screenshot to the screenshot directory with a given name.
    """
    screenshot_path = f'{screenshot_dir}/{name}_{count:04d}.png'
    Image.fromarray(image).save(screenshot_path)
    print(f"Saved screenshot: {screenshot_path}")

def find_best_shift(image1, image2, search_radius=30):
    """
    Finds the best shift that aligns image2 with image1 using template matching
    around a given pixel radius.

    :param image1: Numpy array of the first image.
    :param image2: Numpy array of the second image.
    :param search_radius: Maximum pixel radius to search around the center (default is 19).
    :return: Best shift (shift_x, shift_y) and the maximum similarity score.
    """
    h, w = image1.shape

    best_shift = (0, 0)
    max_corr = -np.inf

    for y_shift in range(-search_radius, search_radius + 1):
        for x_shift in range(-search_radius, search_radius + 1):
            shifted_image = np.roll(image2, shift=(y_shift, x_shift), axis=(0, 1))
            correlation = np.sum(image1 * shifted_image)
            
            if correlation > max_corr:
                max_corr = correlation
                best_shift = (x_shift, y_shift)

    return best_shift, max_corr

def sift_movement(image1, image2, min_match=10):
    """
    Detect movement using SIFT feature matching.

    :param image1: Numpy array of the first image.
    :param image2: Numpy array of the second image.
    :param min_match: Minimum number of good matches required to consider movement detected.
    :return: Tuple of (shift_x, shift_y).
    """
    sift = cv2.SIFT_create()
    kps1, des1 = sift.detectAndCompute(image1, None)
    kps2, des2 = sift.detectAndCompute(image2, None)
    
    if des1 is None or des2 is None:
        return 0, 0
    
    flann = cv2.FlannBasedMatcher(dict(algorithm=0, trees=5), dict(checks=50))
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    
    if len(good_matches) > min_match:
        src_pts = np.float32([kps1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kps2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if M is not None:  # Check if homography was found
            shift_x = M[0, 2]
            shift_y = M[1, 2]
            return shift_x, shift_y

    return 0, 0

def optical_flow_movement(image1, image2, max_offset=19, threshold=5.0, debug=False, screenshot_dir='screenshots', count=0):
    """
    Detect movement using Optical Flow with capped length of pixel offsets and provide debug information.

    :param image1: Numpy array of the first image.
    :param image2: Numpy array of the second image.
    :param max_offset: Maximum pixel offset to cap the flow vectors.
    :param threshold: Minimum magnitude of flow vectors to consider movement detected.
    :param debug: Boolean indicating whether to print debug information and save flow visualization.
    :param screenshot_dir: Directory to save flow visualization if debug is True.
    :param count: Counter for naming the saved screenshots.
    :return: Boolean indicating if significant optical flow was detected.
    """
    # Calculate dense optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(image1, image2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Calculate the magnitude and angle of the flow vectors
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Cap the magnitude to the specified maximum offset
    capped_mag = np.clip(mag, 0, max_offset)

    # Calculate the average magnitude of the capped flow
    avg_magnitude = np.mean(capped_mag)

    if debug:
        print(f"Average Capped Optical Flow Magnitude: {avg_magnitude}")

        # Create a visual representation of the flow
        hsv = np.zeros_like(cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR))
        hsv[..., 1] = 255
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(capped_mag, None, 0, 255, cv2.NORM_MINMAX)
        flow_visual = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Check if there is significant movement
    return avg_magnitude > threshold

def detect_movement(image1, image2,  min_match=10, sift_threshold=5, optical_flow_threshold=5.0, max_offset=19, debug=True, count=0):
    """
    Detect movement by combining SIFT feature matching and Optical Flow strategies.

    :param image1: Numpy array of the first image.
    :param image2: Numpy array of the second image.
    :param min_match: Minimum number of good SIFT matches required to consider movement detected.
    :param sift_threshold: Threshold for shift detection using SIFT.
    :param optical_flow_threshold: Threshold for movement detection using Optical Flow.
    :param max_offset: Maximum pixel offset to cap the flow vectors in Optical Flow.
    :param debug: Boolean indicating whether to print debug information.
    :param count: Counter for naming the saved screenshots.
    :return: Boolean indicating if movement was detected.
    """
    # Use SIFT method to detect movement
    sift_shift = sift_movement(image1, image2, min_match)
    print(f"SIFT detected shift: {sift_shift}")
    
    # Use Optical Flow to detect movement with capped offset and debugging
    optical_flow_detected = optical_flow_movement(image1, image2, max_offset=max_offset, threshold=optical_flow_threshold, debug=debug, screenshot_dir='screenshots', count=count)
    print(f"Optical Flow detected: {optical_flow_detected}")

    # Check if either method detects movement above the threshold
    if (abs(sift_shift[0]) > sift_threshold or abs(sift_shift[1]) > sift_threshold or
        optical_flow_detected):
        return True
    
    return False
