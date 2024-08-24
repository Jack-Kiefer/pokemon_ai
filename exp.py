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
from heapq import heappop, heappush
from menu_checker import MenuDetector  # Assuming you have MenuDetector in a separate file named 'menu_detector.py'

class PokemonExplorer:
    def __init__(self, bbox, interval=.7, pixel_size=(32, 32), resize_factor=0.25, screenshot_dir='screenshots', menu_images_dir='menu_images', hash_threshold=10):
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
        self.current_image = None  # Store the current image for SIFT comparison
        self.screenshot_count = 0  # Counter for the screenshots

        # Initialize MenuDetector
        self.menu_detector = MenuDetector()

    def capture_screenshot(self):
        screenshot = ImageGrab.grab(bbox=self.bbox)
        screenshot = screenshot.resize(
            (int(screenshot.width * self.resize_factor), int(screenshot.height * self.resize_factor)), Image.BILINEAR
        )
        grayscale_screenshot = screenshot.convert('L')
        screenshot_array = np.array(grayscale_screenshot)
        return screenshot_array

    def is_black_or_white_screen(self, image, threshold=20):
        return np.mean(image) < threshold or np.mean(image) > 255 - threshold

    def save_screenshot(self, image, description):
        """
        Save the current screenshot to the screenshot directory with a description.
        
        :param image: The image to save.
        :param description: A string describing the context of the screenshot.
        """
        self.screenshot_count += 1
        screenshot_path = os.path.join(self.screenshot_dir, f'screenshot_{self.screenshot_count:04d}_{description}.png')
        Image.fromarray(image).save(screenshot_path)
        print(f"Screenshot saved: {screenshot_path}")


    def detect_movement_sift(self, image1, image2, min_match=10):
            # Save the images for debugging
        # self.save_screenshot(image1, "sift_image1")
        # self.save_screenshot(image2, "sift_image2")

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

    def detect_movement(self):
        shift_x, shift_y = self.detect_movement_sift(self.prev_image, self.current_image)
        print(shift_x, shift_y)
        return abs(shift_x) > 1 or abs(shift_y) > 1

    def move_direction(self, direction, min_hold_time=0.05, wait_time=0.5):
        keyboard.press(direction)
        time.sleep(min_hold_time)
        keyboard.release(direction)
        time.sleep(wait_time)
        self.current_direction = direction

    def handle_trapped_scenario(self):
        current_pos = self.tile_maps[self.current_tile_map_index].current_position
        neighbors = self.neighbors()

        trapped = all(
            self.tile_maps[self.current_tile_map_index].map[current_pos[0] + dx, current_pos[1] + dy].tile_type == '#'
            for direction, (dx, dy) in neighbors.items()
        )

        if trapped:
            print("Trapped! Removing walls and spamming X to open menu and escape.")
            for direction, (dx, dy) in neighbors.items():
                neighbor_pos = (current_pos[0] + dx, current_pos[1] + dy)
                self.tile_maps[self.current_tile_map_index].map[neighbor_pos[0], neighbor_pos[1]].tile_type = ' '
            
            # Spam 'X' to try to open the menu and validate escape
            while trapped:
                self.current_image = self.capture_screenshot()
                time.sleep(0.2)
                # Try moving in all directions
                for direction in neighbors:
                    self.move_direction(direction)
                    time.sleep(0.5)  # Give some time to see if the movement is successful
                    
                    # Press 'X' to open the menu
                    self.press_key('x')
                    time.sleep(0.5)  # Wait for the menu to potentially open

                    # Check if the menu is open using MenuDetector
                    if self.menu_detector.is_menu_open():
                        print("Escape validated by menu detection.")
                        # Press 'B' to close the menu
                        self.press_key('b')

                        trapped = False
                        print("Moved out of the trap!")
                        # self.find_and_relocate_closest_tile()
                        return
                    self.press_random_key()

                    

                    
    def find_and_relocate_closest_tile(self):
        current_image_hash = imagehash.average_hash(Image.fromarray(self.current_image))
        closest_hash_diff = float('inf')
        closest_position = None
        closest_map_index = None

        hash_threshold = 5  # Define the threshold for hash closeness

        for map_index, tile_map in enumerate(self.tile_maps):
            for x in range(tile_map.width):
                for y in range(tile_map.height):
                    cell_hash = tile_map.map[x, y].hash_value
                    if cell_hash:
                        hash_diff = current_image_hash - cell_hash
                        if hash_diff < closest_hash_diff:
                            closest_hash_diff = hash_diff
                            closest_position = (x, y)
                            closest_map_index = map_index

        if closest_hash_diff <= hash_threshold and closest_position:
            self.current_tile_map_index = closest_map_index
            self.tile_maps[self.current_tile_map_index].set_position(closest_position)
            print(f"Relocated to tile map {self.current_tile_map_index} at position {closest_position} based on hash similarity.")
        else:
            new_tile_map = tilemap.TileMap(100, 100)
            self.tile_maps.append(new_tile_map)
            self.current_tile_map_index = len(self.tile_maps) - 1
            start_position = (50, 50)
            self.tile_maps[self.current_tile_map_index].set_position(start_position)
            print(f"Created new tile map at index {self.current_tile_map_index} due to hash mismatch.")

    def bfs_search(self):
        start_map = self.current_tile_map_index
        start_pos = self.tile_maps[start_map].current_position
        start_direction = self.current_direction

        # Priority queue to manage the search
        queue = []
        heappush(queue, (0, start_map, start_pos, start_direction, []))  # (cost, map, position, direction, path)
        visited = set()

        while queue:
            cost, current_map, current_pos, current_direction, path = heappop(queue)

            if (current_map, current_pos) in visited:
                continue
            visited.add((current_map, current_pos))

            # Check if we reached an unexplored tile
            if self.tile_maps[current_map].map[current_pos[0], current_pos[1]].tile_type == ' ':
                return path

            # Get neighbors and explore
            neighbors = list(self.neighbors().items())

            for direction, (dx, dy) in neighbors:
                neighbor_pos = (current_pos[0] + dx, current_pos[1] + dy)

                # Check if the position is a warp and handle the transition
                if self.is_valid_position(neighbor_pos, self.tile_maps[current_map]):
                    if self.tile_maps[current_map].map[neighbor_pos[0], neighbor_pos[1]].tile_type == 'W':
                        # Handle warp
                        if (current_map, neighbor_pos) in self.warp_dict:
                            warp_map, warp_pos = self.warp_dict[(current_map, neighbor_pos)]
                            # Continue moving in the same direction after warp
                            new_position_after_warp = (warp_pos[0] + dx, warp_pos[1] + dy)
                            if self.is_valid_position(new_position_after_warp, self.tile_maps[warp_map]):
                                turn_cost = 1 if direction != current_direction else 0
                                new_cost = cost + 1 + turn_cost
                                heappush(queue, (new_cost, warp_map, new_position_after_warp, direction, path + [(direction, warp_map)]))
                    else:
                        # Normal movement, no warp
                        turn_cost = 1 if direction != current_direction else 0
                        new_cost = cost + 1 + turn_cost
                        heappush(queue, (new_cost, current_map, neighbor_pos, direction, path + [(direction, current_map)]))

        return []  # No path found

    def neighbors(self):
        return {
            'up': (0, -1), 'down': (0, 1), 'left': (-1, 0), 'right': (1, 0)
        }

    def is_valid_position(self, pos, tile_map):
        x, y = pos
        return 0 <= x < tile_map.width and 0 <= y < tile_map.height and tile_map.map[x, y].tile_type != '#'

    
    def press_key(self, key):
        keyboard.press(key)
        time.sleep(0.05)
        keyboard.release(key)

    def press_random_key(self):
        # Define a list of possible keys to press
        possible_keys = ["up", "left", "right", "down", "a", "b", "a", "b", "a", "b"]
        
        # Randomly select one of the keys
        selected_key = random.choice(possible_keys)
        
        # Press and release the selected key
        keyboard.press(selected_key)
        time.sleep(0.05)
        keyboard.release(selected_key)

        print(f"Pressed key: {selected_key}")  # Optional: Log the pressed key for debugging

        
    def start_battle(self):
        self.current_image = self.capture_screenshot()
        while not self.is_black_or_white_screen(self.current_image):
            time.sleep(0.2)
            self.current_image = self.capture_screenshot()
            self.press_key("a")
    
    def check_for_warp(self):
        # Check for black screen
        start_time = time.time()
        while time.time() - start_time < self.interval:
            self.current_image = self.capture_screenshot()
            if self.is_black_or_white_screen(self.current_image):
                print("Possible warp detected: screen is black.")
                while self.is_black_or_white_screen(self.current_image):
                    self.current_image = self.capture_screenshot()
                time.sleep(1.5)

                # Press 'x' to open the menu
                self.press_key('x')
                time.sleep(0.5)  # Wait for the menu to potentially open

                # Check if the menu is open using MenuDetector
                if self.menu_detector.is_menu_open():
                    print("Warp confirmed by menu detection.")
                    # Press 'b' to close the menu
                    self.press_key('b')

                    # Handle warp detection
                    current_position = self.tile_maps[self.current_tile_map_index].move(self.current_direction)
                    current_key = (self.current_tile_map_index, current_position)

                    if current_key in self.warp_dict:
                        old_index = self.current_tile_map_index
                        self.current_tile_map_index, warp_position = self.warp_dict[current_key]
                        self.tile_maps[self.current_tile_map_index].set_position(warp_position)
                        print(f"Warped to map index {self.current_tile_map_index} at position {warp_position}")

                        self.tile_maps[old_index].map[current_position].tile_type = 'W'
                        self.tile_maps[self.current_tile_map_index].map[warp_position].tile_type = 'W'

                        new_position = self.tile_maps[self.current_tile_map_index].move(self.current_direction)
                        self.tile_maps[self.current_tile_map_index].set_position(new_position)
                    elif any(
                        self.is_valid_position((current_position[0] + dx, current_position[1] + dy), self.tile_maps[self.current_tile_map_index]) and
                        self.tile_maps[self.current_tile_map_index].map[current_position[0] + dx, current_position[1] + dy].tile_type == 'W'
                        for dx, dy in self.neighbors().values()):
                        # Find the neighboring warp tile
                        for (dx, dy) in self.neighbors().values():
                            neighbor_pos = (current_position[0] + dx, current_position[1] + dy)
                            if self.is_valid_position(neighbor_pos, self.tile_maps[self.current_tile_map_index]) and \
                                    self.tile_maps[self.current_tile_map_index].map[neighbor_pos[0], neighbor_pos[1]].tile_type == 'W':
                                
                                # Find the corresponding warp destination from the neighboring warp
                                neighbor_key = (self.current_tile_map_index, neighbor_pos)
                                if neighbor_key in self.warp_dict:
                                    warp_map, warp_pos = self.warp_dict[neighbor_key]
                                    warp_pos_offset = (warp_pos[0] - dx, warp_pos[1] - dy)
                                    # Create a warp between current_position and the offset warp position
                                    self.warp_dict[current_key] = (warp_map, warp_pos_offset)
                                    self.warp_dict[(warp_map, warp_pos_offset)] = current_key

                                    # Mark both positions as warp tiles
                                    self.tile_maps[self.current_tile_map_index].map[current_position[0], current_position[1]].tile_type = 'W'
                                    self.tile_maps[warp_map].map[warp_pos_offset[0], warp_pos_offset[1]].tile_type = 'W'

                                    print(f"Warp created between map {self.current_tile_map_index} at {current_position} and {warp_map} at {warp_pos_offset}")
                                    # Move the player to the new warp position
                                    self.current_tile_map_index = warp_map
                                    self.tile_maps[self.current_tile_map_index].set_position(warp_pos_offset)
                                    new_position = self.tile_maps[self.current_tile_map_index].move(self.current_direction)
                                    self.tile_maps[self.current_tile_map_index].set_position(new_position)
                                    return True
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
                    return True # Skip the rest of the loop after a warp

                else:
                    return False
                    print("No menu detected. Warp is not confirmed.")



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
                # direction = input()
                image = self.capture_screenshot()
                hash = imagehash.average_hash(Image.fromarray(image))

                if not self.tile_maps[self.current_tile_map_index].check_hash(hash):
                    print("Bad hash")

                if direction == 'warp':
                    self.current_tile_map_index = map_index
                    continue

                if self.current_direction != direction:
                    self.move_direction(direction)

                new_position = self.tile_maps[self.current_tile_map_index].move(direction)
                self.move_direction(direction)
                if self.check_for_warp():
                    break

                self.current_image = self.capture_screenshot()
                
                if self.detect_movement():
                    self.tile_maps[self.current_tile_map_index].set_position(new_position)
                    self.tile_maps[self.current_tile_map_index].mark_open(new_position, imagehash.average_hash(Image.fromarray(self.current_image)))
                else:
                    self.tile_maps[self.current_tile_map_index].mark_wall(new_position)
                    self.handle_trapped_scenario()

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
