import numpy as np
import cv2
from PIL import Image
import os
import json
import tensorflow as tf
import time
from PIL import Image
from PIL import ImageGrab, Image

def save_screenshot(image, name, screenshot_dir='screenshots', count=0):
    """
    Save a screenshot to the screenshot directory with a given name.
    """
    if not os.path.exists(screenshot_dir):
        os.makedirs(screenshot_dir)
    
    screenshot_path = f'{screenshot_dir}/{name}_{count:04d}.png'
    Image.fromarray(image).save(screenshot_path)
    print(f"Saved screenshot: {screenshot_path}")
    return screenshot_path

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

def sift_movement(image1, image2, threshold, min_match=10):
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
            return abs(shift_x) > threshold or abs(shift_y) > threshold

    return 0, 0

def optical_flow_movement(image1, image2, max_offset=40, threshold=3.5, debug=False, screenshot_dir='screenshots', count=0):
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

        # # Save the flow visualization if debug is enabled
        # flow_visual_path = save_screenshot(flow_visual, 'flow_visual', screenshot_dir=screenshot_dir, count=count)
        # print(f"Saved flow visualization: {flow_visual_path}")

    # Check if there is significant movement
    return avg_magnitude > threshold

# def detect_movement(image1, image2, model, min_match=10, sift_threshold=3, optical_flow_threshold=4, max_offset=19, debug=False):
#     """
#     Detect movement using traditional computer vision methods and ML prediction.
    
#     :param image1: Numpy array of the first image.
#     :param image2: Numpy array of the second image.
#     :param model: Preloaded ML model to predict movement.
#     :param min_match: Minimum matches for SIFT.
#     :param sift_threshold: SIFT movement detection threshold.
#     :param optical_flow_threshold: Optical Flow movement detection threshold.
#     :param max_offset: Maximum offset for Optical Flow.
#     :param debug: Debug mode for additional output.
#     :return: Boolean indicating if movement was detected.
#     """

#     # Ensure images are in grayscale for SIFT and Optical Flow
#     if len(image1.shape) == 3:
#         image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
#     else:
#         image1_gray = image1

#     if len(image2.shape) == 3:
#         image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
#     else:
#         image2_gray = image2

#     # Use traditional methods to detect movement
#     sift_shift = sift_movement(image1_gray, image2_gray, min_match)
#     optical_flow_detected = optical_flow_movement(image1_gray, image2_gray, max_offset=max_offset, threshold=optical_flow_threshold, debug=debug)

#     # Preprocess images for the ML model
#     img1_resized = cv2.resize(image1, (128, 128)).astype('float32') / 255.0
#     img2_resized = cv2.resize(image2, (128, 128)).astype('float32') / 255.0
#     img1_resized = np.expand_dims(img1_resized, axis=0)
#     img2_resized = np.expand_dims(img2_resized, axis=0)
#     combined_images = np.concatenate([img1_resized, img2_resized], axis=-1)

#     # Make a prediction
#     ml_detected = model.predict(combined_images)[0, 0] > 0.5

#     # Combine results
#     movement_detected = ml_detected or (abs(sift_shift[0]) > sift_threshold or abs(sift_shift[1]) > sift_threshold or optical_flow_detected)

#     return movement_detected

def prepare_images(image1, image2):
    # Resize images and ensure they are in RGB format (three channels)
    img1_resized = cv2.resize(image1, (128, 128)).astype('float32') / 255.0
    img2_resized = cv2.resize(image2, (128, 128)).astype('float32') / 255.0

    # Ensure both images have three channels if they are grayscale
    if img1_resized.ndim == 2:
        img1_resized = cv2.cvtColor(img1_resized, cv2.COLOR_GRAY2RGB)
    if img2_resized.ndim == 2:
        img2_resized = cv2.cvtColor(img2_resized, cv2.COLOR_GRAY2RGB)

    # Concatenate images along the channel axis to form an input with 6 channels
    combined_images = np.concatenate([img1_resized, img2_resized], axis=2)

    # Add batch dimension
    combined_images = np.expand_dims(combined_images, axis=0)
    return combined_images

def detect_movement(image1, image2, model, count, min_match=10, sift_threshold=5, optical_flow_threshold=4.5, max_offset=19, debug=False):
    """
    Use a pre-trained ML model to predict movement between two images.

    :param image1: Numpy array of the first RGB image.
    :param image2: Numpy array of the second RGB image.
    :param model: Pre-trained ML model for movement detection.
    :return: Integer indicating movement category: 0 (no movement), 1 (movement detected), or 2 (ledge).
    """
    # save_screenshot(image1, 'image1', count=count)
    # save_screenshot(image2, 'image2', count=count)

    # Prepare images for the model
    input_data = prepare_images(image1, image2)

    # Make a prediction
    prediction = model.predict(input_data)

    # Get the index of the class with the highest probability
    predicted_class = np.argmax(prediction, axis=1)[0]

    return predicted_class
