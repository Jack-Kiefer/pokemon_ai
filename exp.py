import os
import numpy as np
from PIL import ImageGrab, Image
import time
import pydirectinput
import keyboard
import cv2
import tilemap  # Assuming you have a tilemap module
from collections import deque
import random
import imagehash

class PokemonExplorer:
    def __init__(self, bbox, interval=.6, pixel_size=(32, 32), resize_factor=0.25, screenshot_dir='screenshots'):
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
        self.warp_dict = {}  # Two-way dictionary to map warp points
        self.prev_image = None  # Store the previous image for SIFT comparison
        self.current_image = None  # Store the previous image for SIFT comparison
        self.screenshot_count = 0  # Counter for the screenshots

    def capture_screenshot(self):
        screenshot = ImageGrab.grab(bbox=self.bbox)
        screenshot = screenshot.resize(
            (int(screenshot.width * self.resize_factor), int(screenshot.height * self.resize_factor)), Image.BILINEAR
        )
        grayscale_screenshot = screenshot.convert('L')
        screenshot_array = np.array(grayscale_screenshot)

        # # Save the screenshot
        # screenshot_path = os.path.join(self.screenshot_dir, f'screenshot_{self.screenshot_count:04d}.png')
        # grayscale_screenshot.save(screenshot_path)
        # print(f"Saved screenshot: {screenshot_path}")
        # self.screenshot_count += 1

        return screenshot_array

    def is_black_or_white_screen(self, image, threshold=20):
        # Check if the image is mostly black
        return np.mean(image) < threshold or np.mean(image) > 255 - threshold

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

            if M is not None:  # Check if homography was found
                shift_x = M[0, 2]
                shift_y = M[1, 2]
                return shift_x, shift_y
            else:
                print("Homography could not be computed. No significant movement detected.")
                return 0, 0

        print("Not enough good matches found. No significant movement detected.")
        return 0, 0


    def move_direction(self, direction, min_hold_time=0.05, wait_time=0.5):
        keyboard.press(direction)
        time.sleep(min_hold_time)
        keyboard.release(direction)
        time.sleep(wait_time)
        self.current_direction = direction


    def bfs_search(self):
        start_map = self.current_tile_map_index
        start_pos = self.tile_maps[start_map].current_position

        queue = deque([(start_map, start_pos, [])])
        visited = set()

        while queue:
            current_map, current_pos, path = queue.popleft()

            if (current_map, current_pos) in visited:
                continue
            visited.add((current_map, current_pos))

            # Check if we reached an unexplored tile
            if self.tile_maps[current_map].map[current_pos[0], current_pos[1]].tile_type == ' ':
                return path

            # Get neighbors and shuffle the order of exploration
            neighbors = list(self.neighbors().items())
            random.shuffle(neighbors)  # Shuffle the directions

            # Explore neighbors
            for direction, (dx, dy) in neighbors:
                neighbor_pos = (current_pos[0] + dx, current_pos[1] + dy)

                if self.is_valid_position(neighbor_pos, self.tile_maps[current_map]):
                    queue.append((current_map, neighbor_pos, path + [(direction, current_map)]))

            # Handle warps
            if (current_map, current_pos) in self.warp_dict:
                warp_map, warp_pos = self.warp_dict[(current_map, current_pos)]
                neighbors = list(self.neighbors().items())
                random.shuffle(neighbors)  # Shuffle the directions for warp as well

                for direction, (dx, dy) in neighbors:
                    new_position_after_warp = (warp_pos[0] + dx, warp_pos[1] + dy)
                    if self.is_valid_position(new_position_after_warp, self.tile_maps[warp_map]):
                        # Add the direction to the path and immediately move one step after the warp
                        queue.append((warp_map, new_position_after_warp, path + [('warp', warp_map), (direction, warp_map)]))

        return []  # No path found



    def neighbors(self):
        return {
            'up': (0, -1), 'down': (0, 1), 'left': (-1, 0), 'right': (1, 0)
        }

    def is_valid_position(self, pos, tile_map):
        x, y = pos
        return 0 <= x < tile_map.width and 0 <= y < tile_map.height and tile_map.map[x, y].tile_type != '#'
    
    def start_battle(self):
        self.current_image = self.capture_screenshot()
        while not self.is_black_or_white_screen(self.current_image):
            time.sleep(0.2)
            self.current_image = self.capture_screenshot()
            keyboard.press("a")
            time.sleep(.05)
            keyboard.release("a")
    
    def process_warp_or_battle(self):
        #check for black screen
        start_time = time.time()
        while(time.time() - start_time < self.interval):
                self.current_image = self.capture_screenshot()
                if self.is_black_or_white_screen(self.current_image):
                    print("Warp detected: screen is black.")
                    while self.is_black_or_white_screen(self.current_image):

                        self.current_image = self.capture_screenshot()

                    warp = True
                    start_time = time.time()
                    while (time.time() - start_time < .5):
                        self.current_image = self.capture_screenshot()
                        if self.is_black_or_white_screen(self.current_image):
                            print("battle!")
                            time.sleep(2)
                            self.start_battle()
                            time.sleep(2)
                            warp = False
                            
                    if (not warp):
                        break
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
                        self.tile_maps[old_index].map[current_position].tile_type = 'W'
                        self.tile_maps[self.current_tile_map_index].map[warp_position].tile_type = 'W'
                        print(f"Created new tile map at index {self.current_tile_map_index}")
                        
                    self.prev_image = self.capture_screenshot()
                    self.tile_maps[self.current_tile_map_index].print_map()
                    return  # Skip the rest of the loop after a warp



    def explore(self):
        time.sleep(1)  # Pause for 1 second on startup
        self.prev_image = self.capture_screenshot()

        while True:
            # Use BFS to find the path to the nearest unexplored area
            path = self.bfs_search()
            if not path:
                print("No unexplored tiles found! Exploration complete.")
                break

            for direction, map_index in path:
                image = self.capture_screenshot()
                hash = imagehash.average_hash(Image.fromarray(image))

                if (not self.tile_maps[self.current_tile_map_index].check_hash(hash)):
                    print("bad hash")

                if direction == 'warp':
                    self.current_tile_map_index = map_index
                    continue

                if self.current_direction != direction:
                    self.move_direction(direction)
                    self.process_warp_or_battle()

                new_position = self.tile_maps[self.current_tile_map_index].move(direction)
                self.move_direction(direction)

                self.process_warp_or_battle()
        
                # Detect movement between frames
                shift_x, shift_y = self.detect_movement_sift(self.prev_image, self.current_image)
                if (abs(shift_x) > 5 or abs(shift_y) > 5):
                    # if abs(shift_x) > abs(shift_y):
                    #     if shift_x > 5:
                    #         print(f"Detected movement to the left: x_shift={shift_x}")
                    #     elif shift_x < -5:
                    #         print(f"Detected movement to the right: x_shift={shift_x}")
                    # else:
                    #     if shift_y > 5:
                    #         print(f"Detected movement up: y_shift={shift_y}")
                    #     elif shift_y < -5:
                    #         print(f"Detected movement down: y_shift={shift_y}")
                    self.tile_maps[self.current_tile_map_index].set_position(new_position)
                    self.tile_maps[self.current_tile_map_index].mark_open(new_position, imagehash.average_hash(Image.fromarray(self.current_image)))
                else:
                    # print(f"No significant movement detected.  x_shift={shift_x}, y_shift={shift_y}")
                    self.tile_maps[self.current_tile_map_index].mark_wall(new_position)

                self.prev_image = self.current_image
                self.tile_maps[self.current_tile_map_index].print_map()

    def get_direction(self, current_pos, next_pos):
        if next_pos[0] > current_pos[0]:
            return 'right'
        elif next_pos[0] < current_pos[0]:
            return 'left'
        elif next_pos[1] > current_pos[1]:
            return 'down'
        elif next_pos[1] < current_pos[1]:
            return 'up'
        return self.current_direction  # Default to current direction

# Example usage
bbox = (1275, 135, 2565, 1100)  # Coordinates for the region you want to capture
explorer = PokemonExplorer(bbox=bbox)
explorer.explore()
