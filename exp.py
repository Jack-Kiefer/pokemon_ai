import numpy as np
from PIL import ImageGrab, Image
import time
import pydirectinput
import imagehash
import keyboard
import random
from collections import deque
import cv2

class TileMap:
    def __init__(self, width, height):
        self.map = np.full((width, height), ' ')  # Initialize an empty map with ' ' as unexplored
        self.width = width
        self.height = height
        self.current_position = (width // 2, height // 2)  # Start at the center
        self.current_direction = 'left'  # Start facing 'left'
        self.min_x, self.min_y = self.current_position  # Bounds of explored area
        self.max_x, self.max_y = self.current_position

        # Mark the starting position as explored
        self.map[self.current_position] = '.'

    def reset(self):
        self.map = np.full((self.width, self.height), ' ')  # Reset the map
        self.current_position = (self.width // 2, self.height // 2)  # Reset to the center
        self.current_direction = 'left'  # Reset facing direction
        self.min_x, self.min_y = self.current_position  # Reset bounds
        self.max_x, self.max_y = self.current_position
        self.map[self.current_position] = '.'  # Mark starting position as explored

    def move(self, direction):
        x, y = self.current_position
        if direction == 'up':
            return (x, y - 1)
        elif direction == 'down':
            return (x, y + 1)
        elif direction == 'left':
            return (x - 1, y)
        elif direction == 'right':
            return (x + 1, y)

    def mark_wall(self, position):
        self.map[position] = '#'
        self.update_bounds(position)

    def mark_open(self, position):
        self.map[position] = '.'
        self.update_bounds(position)

    def set_position(self, position):
        self.current_position = position
        self.map[position] = '.'  # Mark the AI's current position as open space
        self.update_bounds(position)

    def set_direction(self, direction):
        self.current_direction = direction

    def update_bounds(self, position):
        x, y = position
        self.min_x = min(self.min_x, x)
        self.max_x = max(self.max_x, x)
        self.min_y = min(self.min_y, y)
        self.max_x = max(self.max_x, x)
        self.max_y = max(self.max_y, y)

    def print_map(self):
        # Temporarily mark the current position as 'X' for printing
        temp_map = np.copy(self.map)
        x, y = self.current_position
        temp_map[x, y] = 'X'
        
        print("\nCurrent Map:")
        for y in range(self.min_y, self.max_y + 1):
            row = "".join(temp_map[x][y] for x in range(self.min_x, self.max_x + 1))
            print(row)
        print()

    def find_nearest_unexplored(self):
        # Use BFS to find the nearest unexplored tile (' ')
        start = self.current_position
        queue = deque([(start, [])])  # Queue holds tuples of (position, path_to_position)
        visited = set()

        while queue:
            (x, y), path = queue.popleft()
            if (x, y) in visited:
                continue
            visited.add((x, y))

            if self.map[x, y] == ' ':
                return path  # Return the path to the nearest unexplored tile
            directions = ['up', 'left', 'right', 'down']
            random.shuffle(directions)
            d = {
                'up': (0, -1), 'down': (0, 1), 'left': (-1, 0), 'right': (1, 0)
            }
            # Explore neighbors
            for direction in directions:
                new_position = (x + d[direction][0], y + d[direction][1])
                if 0 <= new_position[0] < self.width and 0 <= new_position[1] < self.height:
                    if new_position not in visited and self.map[new_position] != '#':
                        queue.append((new_position, path + [direction]))

        return []  # No unexplored tiles found

class PokemonExplorer:
    def __init__(self, bbox, interval=0.4, pixel_size=(32, 32), hash_size=8, hash_threshold=5, resize_factor = .25):
        self.bbox = bbox
        self.interval = interval
        self.pixel_size = pixel_size
        self.hash_size = hash_size
        self.hash_threshold = hash_threshold
        self.tile_map = TileMap(100, 100)  # Create a 100x100 map
        self.prev_image = None  # Store the previous image for SIFT comparison
        self.resize_factor = resize_factor

    def capture_screenshot(self):
        screenshot = ImageGrab.grab(bbox=self.bbox)
        return screenshot.resize(self.pixel_size, Image.BILINEAR)

    def detect_movement_sift(self, image1, image2, min_match=10):
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
        
    def capture_screenshot(self, resize_factor=0.25):
        screenshot = ImageGrab.grab(bbox=self.bbox)
        screenshot = screenshot.resize(
            (int(screenshot.width * resize_factor), int(screenshot.height * resize_factor)), Image.BILINEAR
        )
        grayscale_screenshot = screenshot.convert('L')
        return np.array(grayscale_screenshot)

    def perform_action(self, action):
        actions = {'up': 'up', 'down': 'down', 'left': 'left', 'right': 'right'}
        pydirectinput.press(actions[action], interval=.1)

    def press_random_button(self):
        button = random.choice(['a', 'b'])
        pydirectinput.press(button)

    def turn_direction(self, direction, min_hold_time=0.05, wait_time=.5):
        keyboard.press(direction)   # Press and hold the key down
        time.sleep(min_hold_time)   # Hold it for the minimum time
        keyboard.release(direction) # Release the key
        time.sleep(wait_time)   # Small delay before the next key press

    def explore(self):
        time.sleep(1)  # Pause for 1 second on startup

        while True:  # Loop indefinitely to continue exploring
            path = self.tile_map.find_nearest_unexplored()
            if not path:
                print("No unexplored tiles found! Resetting map...")
                self.tile_map.reset()  # Reset the tilemap when trapped
                self.prev_image = None  # Reset the previous image
                continue  # Restart the exploration

            for direction in path:
                if self.tile_map.current_direction != direction:
                    # If the direction is different, first turn towards it
                    self.turn_direction(direction)
                    self.tile_map.set_direction(direction)

                # Compute the new position
                new_position = self.tile_map.move(direction)

                # Perform the movement in the desired direction
                self.turn_direction(direction)

               
                current_image =  self.capture_screenshot()

                # If this is the first image, set it as the previous and continue
                if self.prev_image is None:
                    self.prev_image = current_image
                    continue

                # Detect movement using SIFT
                shift_x, shift_y, _ = self.detect_movement_sift(self.prev_image, current_image)

                # Check if the current movement contradicts previous knowledge
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
                    self.tile_map.set_position(new_position)
                    self.tile_map.mark_open(new_position)
                else:
                    print(f"No significant movement detected.  x_shift={shift_x}, y_shift={shift_y}")
                    self.tile_map.mark_wall(new_position)

                # Update the previous image
                self.prev_image = current_image

                # Print the map after every move
                self.tile_map.print_map()

                # Press A or B randomly after each movement
                self.press_random_button()

# Example usage
bbox = (1250, 110, 2600, 1100)  # Coordinates for the region you want to capture
explorer = PokemonExplorer(bbox=bbox)
explorer.explore()
