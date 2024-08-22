import numpy as np
from PIL import ImageGrab, Image
import time
import pydirectinput
import keyboard
import random
import cv2
import tilemap  # Assuming you have a tilemap module

class PokemonExplorer:
    def __init__(self, bbox, interval=0.4, pixel_size=(32, 32), resize_factor=0.25):
        self.bbox = bbox
        self.interval = interval
        self.pixel_size = pixel_size
        self.resize_factor = resize_factor

        # List of all tile maps
        self.tile_maps = [tilemap.TileMap(100, 100)]  # Start with one tile map
        self.current_direction = 'left'
        self.current_tile_map_index = 0  # Index of the current tile map
        self.warp_dict = {}  # Dictionary to map warp points to destination tile maps and coordinates
        self.prev_image = None  # Store the previous image for SIFT comparison

    def capture_screenshot(self):
        screenshot = ImageGrab.grab(bbox=self.bbox)
        screenshot = screenshot.resize(
            (int(screenshot.width * self.resize_factor), int(screenshot.height * self.resize_factor)), Image.BILINEAR
        )
        grayscale_screenshot = screenshot.convert('L')
        return np.array(grayscale_screenshot)

    def is_black_screen(self, image, threshold=20):
        # Check if the image is mostly black
        return np.mean(image) < threshold

    def detect_movement_sift(self, image1, image2, min_match=10):
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

            shift_x = M[0, 2]
            shift_y = M[1, 2]
            return shift_x, shift_y

        return 0, 0

    def press_random_button(self):
        button = random.choice(['a', 'b'])
        pydirectinput.press(button)

    def move_direction(self, direction, min_hold_time=0.05, wait_time=0.5):
        keyboard.press(direction)
        time.sleep(min_hold_time)
        keyboard.release(direction)
        time.sleep(wait_time)
        self.current_direction = direction

    def manual_explore(self):
        time.sleep(1)  # Pause for 1 second on startup
        self.prev_image = self.capture_screenshot()

        while True:
            # Get direction input from the user
            direction = input("Enter direction (up, down, left, right) or 'warp' to simulate a warp, or 'quit' to exit: ").strip().lower()
            if direction == 'quit':
                break

            if direction == 'warp':
                # Simulate a warp
                current_position = self.tile_maps[self.current_tile_map_index].current_position

                # Check if this warp point is already known
                if current_position in self.warp_dict:
                    self.current_tile_map_index, warp_position = self.warp_dict[current_position]
                    
                    print(f"Warped to map index {self.current_tile_map_index} at position {warp_position}")

                    new_position = self.tile_maps[self.current_tile_map_index].move(self.current_direction)

                    self.tile_maps[self.current_tile_map_index].set_position(new_position)

                else:
                    # Create a new tile map for the warp destination
                    new_tile_map = tilemap.TileMap(100, 100)
                    self.tile_maps.append(new_tile_map)
                    old_index = self.current_tile_map_index
                    self.current_tile_map_index = len(self.tile_maps) - 1
                    warp_position = (50, 50)  # Default starting position in the new tile map
                    self.tile_maps[self.current_tile_map_index].set_position(warp_position)
                    
                    new_position = self.tile_maps[self.current_tile_map_index].move(self.current_direction)
                    self.tile_maps[self.current_tile_map_index].set_position(new_position)
                    
                    # Record this warp connection
                    self.warp_dict[current_position] = (self.current_tile_map_index, warp_position)
                    self.warp_dict[warp_position] = (old_index, current_position)
                    self.tile_maps[old_index].map[current_position] = 'W'
                    self.tile_maps[self.current_tile_map_index].map[warp_position] = 'W'  # Label the new warp point
                    print(f"Created new tile map at index {self.current_tile_map_index}")
                self.prev_image = self.capture_screenshot()
                self.tile_maps[self.current_tile_map_index].print_map()
                continue  # Skip the rest of the loop after a warp

            if direction not in ['up', 'down', 'left', 'right']:
                print("Invalid direction. Please enter 'up', 'down', 'left', 'right', 'warp', or 'quit'.")
                continue

            if self.current_direction != direction:
                self.move_direction(direction)

            new_position = self.tile_maps[self.current_tile_map_index].move(direction)
            self.move_direction(direction)

            current_image = self.capture_screenshot()

           
            shift_x, shift_y = self.detect_movement_sift(self.prev_image, current_image)
            if (abs(shift_x) > 5 or abs(shift_y) > 5):
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
                self.tile_maps[self.current_tile_map_index].set_position(new_position)
                self.tile_maps[self.current_tile_map_index].mark_open(new_position)
            else:
                print(f"No significant movement detected.  x_shift={shift_x}, y_shift={shift_y}")
                self.tile_maps[self.current_tile_map_index].mark_wall(new_position)

            self.prev_image = current_image

            self.tile_maps[self.current_tile_map_index].print_map()

# Example usage
bbox = (1250, 110, 2600, 1100)  # Coordinates for the region you want to capture
explorer = PokemonExplorer(bbox=bbox)
explorer.manual_explore()
