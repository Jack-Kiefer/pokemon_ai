import os
import numpy as np
from PIL import ImageGrab, Image
import tensorflow as tf
import time
import cv2
from sklearn.preprocessing import LabelEncoder

class TileClassifier:
    def __init__(self, bbox, model_path, tile_size=(32, 32), grid_size=(16, 16), resize_factor=0.25, screenshot_dir='screenshots'):
        self.bbox = bbox
        self.tile_size = tile_size
        self.grid_size = grid_size
        self.resize_factor = resize_factor
        self.screenshot_dir = screenshot_dir
        self.model = tf.keras.models.load_model(model_path)
        self.mental_map = np.zeros((grid_size[0], grid_size[1]), dtype=int)  # Initialize a mental map of size 16x16 with all zeros

        # Create the screenshots directory if it doesn't exist
        os.makedirs(self.screenshot_dir, exist_ok=True)

        # Label encoding for the classes (walkable, not walkable, ledge)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(["walkable", "not_walkable", "ledge"])

    def capture_screenshot(self):
        """
        Capture a screenshot of the defined bounding box, resize it, and return as a numpy array.
        """
        screenshot = ImageGrab.grab(bbox=self.bbox)
        screenshot = screenshot.resize(
            (int(screenshot.width * self.resize_factor), int(screenshot.height * self.resize_factor)), Image.BILINEAR
        )
        return np.array(screenshot)

    def process_tile(self, tile_image):
        """
        Preprocess the tile image to be fed into the model.
        """
        tile_image = cv2.resize(tile_image, self.tile_size)
        tile_image = tile_image.astype('float32') / 255.0  # Normalize to [0, 1]
        tile_image = np.expand_dims(tile_image, axis=0)  # Add batch dimension
        return tile_image

    def classify_tile(self, tile_image):
        """
        Use the trained model to classify the tile.
        """
        processed_tile = self.process_tile(tile_image)
        prediction = self.model.predict(processed_tile)
        predicted_class = np.argmax(prediction, axis=1)
        predicted_label = self.label_encoder.inverse_transform(predicted_class)
        return predicted_label[0]

    def update_mental_map(self, grid_x, grid_y, classification):
        """
        Update the mental map with the classification result.
        """
        class_to_int = {"walkable": 1, "not_walkable": -1, "ledge": 2}
        self.mental_map[grid_x, grid_y] = class_to_int[classification]

    def explore_and_classify(self):
        """
        Main loop to capture screenshot, split into tiles, classify each tile, and update the mental map.
        """
        screenshot = self.capture_screenshot()
        height, width, _ = screenshot.shape
        tile_height = height // self.grid_size[0]
        tile_width = width // self.grid_size[1]

        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                # Extract the tile
                x_start = j * tile_width
                y_start = i * tile_height
                x_end = x_start + tile_width
                y_end = y_start + tile_height
                tile = screenshot[y_start:y_end, x_start:x_end]

                # Classify the tile
                classification = self.classify_tile(tile)

                # Update the mental map
                self.update_mental_map(i, j, classification)

        # Display the current mental map
        self.display_mental_map()

    def display_mental_map(self):
        """
        Display the mental map in a human-readable form.
        """
        display_map = self.mental_map.copy()
        display_map[display_map == 1] = 0  # Walkable
        display_map[display_map == -1] = 255  # Not Walkable
        display_map[display_map == 2] = 128  # Ledge

        display_image = Image.fromarray(display_map.astype(np.uint8))
        display_image = display_image.resize((self.grid_size[0] * 32, self.grid_size[1] * 32), Image.NEAREST)
        display_image.show()

# Example usage
bbox = (1275, 135, 2565, 1100)  # Coordinates for the region you want to capture
model_path = 'tile_classification_model.h5'  # Path to the trained model
classifier = TileClassifier(bbox, model_path)
classifier.explore_and_classify()
