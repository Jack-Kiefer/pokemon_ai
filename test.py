import os
import numpy as np
from PIL import ImageGrab, Image
import time
import pydirectinput
import keyboard
import cv2
import tilemap  # Assuming you have a tilemap module
from collections import deque

class PokemonExplorer:
    def __init__(self, bbox, interval=.4, pixel_size=(32, 32), resize_factor=0.25, screenshot_dir='screenshots'):
        self.bbox = bbox
        self.interval = interval
        self.pixel_size = pixel_size
        self.resize_factor = resize_factor
        self.screenshot_dir = screenshot_dir

        # Create the screenshots directory if it doesn't exist
        os.makedirs(self.screenshot_dir, exist_ok=True)

        # List of all tile maps
        self.tile_maps = [tilemap.TileMap(100, 100)]  # Start with one tile map
        self.current_direction = 'left'
        self.current_tile_map_index = 0  # Index of the current tile map
        self.warp_dict = {}  # Dictionary to map warp points to destination tile maps and coordinates
        self.prev_image = None  # Store the previous image for SIFT comparison
        self.screenshot_count = 0  # Counter for the screenshots

    def capture_screenshot(self):
        screenshot = ImageGrab.grab(bbox=self.bbox)
        screenshot = screenshot.resize(
            (int(screenshot.width * self.resize_factor), int(screenshot.height * self.resize_factor)), Image.BILINEAR
        )
        grayscale_screenshot = screenshot.convert('L')
        screenshot_array = np.array(grayscale_screenshot)

        # Save the screenshot
        screenshot_path = os.path.join(self.screenshot_dir, f'screenshot_{self.screenshot_count:04d}.png')
        grayscale_screenshot.save(screenshot_path)
        print(f"Saved screenshot: {screenshot_path}")
        self.screenshot_count += 1

        return screenshot_array

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

    def find_nearest_unexplored(self):
        # Use BFS to find the nearest unexplored tile (' ') or warp point ('W')
        start = self.tile_maps[self.current_tile_map_index].current_position
        queue = deque([(start, [])])  # Queue holds tuples of (position, path_to_position)
        visited = set()

        while queue:
            (x, y), path = queue.popleft()
            if (x, y) in visited:
                continue
            visited.add((x, y))

            # Prioritize unexplored tiles first
            if self.tile_maps[self.current_tile_map_index].map[x, y] == ' ':
                return path  # Return the path to the nearest unexplored tile

            # Consider warp points if no unexplored tiles are found
            if self.tile_maps[self.current_tile_map_index].map[x, y] == 'W':
                return path  # Return the path to the nearest warp point

            directions = {
                'up': (0, -1), 'down': (0, 1), 'left': (-1, 0), 'right': (1, 0)
            }
            # Explore neighbors
            for direction, (dx, dy) in directions.items():
                new_position = (x + dx, y + dy)
                if 0 <= new_position[0] < self.tile_maps[self.current_tile_map_index].width and 0 <= new_position[1] < self.tile_maps[self.current_tile_map_index].height:
                    if new_position not in visited and self.tile_maps[self.current_tile_map_index].map[new_position] != '#':
                        queue.append((new_position, path + [direction]))

        return []  # No unexplored tiles or warp points found


    def explore(self):
        time.sleep(1)  # Pause for 1 second on startup
        self.prev_image = self.capture_screenshot()

        while True:
            # Find the path to the nearest unexplored area
            path = self.find_nearest_unexplored()
            if not path:
                print("No unexplored tiles found! Exploration complete.")
                break

            for direction in path:
                if self.current_direction != direction:
                    self.move_direction(direction)

                new_position = self.tile_maps[self.current_tile_map_index].move(direction)
                self.move_direction(direction)
                time.sleep(self.interval)

                current_image = self.capture_screenshot()

                # Check for warp detection by detecting black screen
                if self.is_black_screen(current_image):
                    print("Warp detected: screen is black.")
                    while self.is_black_screen(current_image):
                        time.sleep(0.5)
                        current_image = self.capture_screenshot()
                    time.sleep(0.5)

                    # Handle warp detection
                    current_position = self.tile_maps[self.current_tile_map_index].move(self.current_direction)
                    current_key = (self.current_tile_map_index, current_position)
                    
                    if current_key in self.warp_dict:
                        self.current_tile_map_index, warp_position = self.warp_dict[current_key]
                        self.tile_maps[self.current_tile_map_index].set_position(warp_position)
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

                        # Record this warp connection in both directions
                        new_key = (self.current_tile_map_index, warp_position)
                        self.warp_dict[current_key] = new_key
                        self.warp_dict[new_key] = current_key

                        # Label the warp points on the maps
                        self.tile_maps[old_index].map[current_position] = 'W'
                        self.tile_maps[self.current_tile_map_index].map[warp_position] = 'W'
                        print(f"Created new tile map at index {self.current_tile_map_index}")
                        
                    self.prev_image = self.capture_screenshot()
                    self.tile_maps[self.current_tile_map_index].print_map()
                    continue  # Skip the rest of the loop after a warp

                # Detect movement between frames
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
explorer.explore()
