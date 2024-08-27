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
from movement_detector import detect_movement
import signal
import json
import tensorflow as tf

class PokemonExplorer:
    def __init__(self, bbox, model, interval=2.5, pixel_size=(32, 32), resize_factor=0.25, screenshot_dir='screenshots', menu_images_dir='menu_images', hash_threshold=20):
        self.bbox = bbox
        self.model = model
        self.interval = interval
        self.pixel_size = pixel_size
        self.resize_factor = resize_factor
        self.screenshot_dir = screenshot_dir

        # Create the screenshots directory if it doesn't exist
        os.makedirs(self.screenshot_dir, exist_ok=True)

        # List of all tile maps
        self.tile_maps = [tilemap.TileMap(800, 800)]  # Start with one tile map
        self.current_direction = 'up'
        self.current_tile_map_index = 0  # Index of the current tile map
        self.warp_dict = {}  # Two-way dictionary to map warp points
        self.prev_image = None  # Store the previous image for SIFT comparison
        self.prev_hash = None
        self.current_image = None  # Store the current image for SIFT comparison
        self.screenshot_count = 0  # Counter for the screenshots
        self.counter = 2932

        # Initialize MenuDetector
        self.menu_detector = MenuDetector()

             # Set up signal handler for graceful exit
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, sig, frame):
        """
        Signal handler for SIGINT (Ctrl+C).
        Saves the tile maps to a file before exiting.
        """
        print("\nCtrl+C detected. Saving tile maps to file...")
        self.save_tile_maps_to_text_files()
        print("Tile maps saved. Exiting...")
        exit(0)

    def save_tile_maps_to_text_files(self, directory='tile_maps_text'):
        """
        Saves each tile map to individual text files, representing the entire explored area.
        """
        # process_images(self.image_pairs)

        

        for index, tile_map in enumerate(self.tile_maps):
            file_path = os.path.join(directory, f'tile_map_{index:04d}.txt')
            with open(file_path, 'w') as file:
                # Define the boundaries of the explored area
                min_x = tile_map.min_x
                max_x = tile_map.max_x
                min_y = tile_map.min_y
                max_y = tile_map.max_y

                # Temporarily mark the current position as 'X' for saving
                temp_map = np.full((tile_map.width, tile_map.height), ' ')
                for i in range(min_x, max_x + 1):
                    for j in range(min_y, max_y + 1):
                        temp_map[i, j] = tile_map.map[i, j].tile_type

                # Mark the current position with 'X'
                x, y = tile_map.current_position
                temp_map[x, y] = 'X'

                file.write(f"Tile Map {index} (explored area):\n")
                for j in range(min_y, max_y + 1):
                    row = "".join(temp_map[i][j] for i in range(min_x, max_x + 1))
                    file.write(row + "\n")
            
            print(f"Tile map {index} saved as {file_path}")


    def capture_screenshot(self):
        screenshot = ImageGrab.grab(bbox=self.bbox)
        screenshot = screenshot.resize(
            (int(screenshot.width * self.resize_factor), int(screenshot.height * self.resize_factor)), Image.BILINEAR
        )
        grayscale_screenshot = screenshot.convert('L')
        screenshot_array = np.array(grayscale_screenshot)
        return screenshot_array

    def is_black_or_white_screen(self, image, threshold=10):
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

    def move_direction(self, direction, min_hold_time=0.05, wait_time=0.40):
        keyboard.press(direction)
        time.sleep(min_hold_time)
        keyboard.release(direction)
        time.sleep(wait_time)
        self.current_direction = direction

    def turn_direction(self, direction, min_hold_time=0.05, wait_time=0.05):
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
            
            self.spam_until_escape()

    def spam_until_escape(self):
        while True:
            if (not self.in_cutscene()):
                return 
            self.press_random_key()
    
    def in_cutscene(self):
        # Press 'X' to open the menu
        self.press_key('x')
        time.sleep(.05)
        # Check if the menu is open using MenuDetector
        if self.menu_detector.is_menu_open():
            self.press_key('b')
            time.sleep(.05)
            return False
        return True

                    
    def find_and_relocate_closest_tile(self, relocation_threshold=10):
        """
        Find and relocate to the closest tile based on hash similarity.
        Relocate if the current tile hash is very different from the screenshot hash.

        :param relocation_threshold: Threshold for hash difference to trigger relocation (default is 10).
        """
        self.current_image = self.capture_screenshot()
        current_image_hash = imagehash.average_hash(Image.fromarray(self.current_image))
        
        # Get the current tile and its hash
        current_x, current_y = self.tile_maps[self.current_tile_map_index].current_position
        current_tile = self.tile_maps[self.current_tile_map_index].map[(current_x, current_y)]
        current_tile_hash = current_tile.image_hash if current_tile else None

        # Check if relocation is necessary
        if current_tile_hash != None and current_image_hash - current_tile_hash > relocation_threshold:
            print("Current tile hash is significantly different. Searching for a closer match.")
            
            closest_hash_diff = float('inf')
            closest_position = None
            closest_map_index = None

            # Search for the closest matching tile across all tile maps
            for i in range(len(self.tile_maps)):
                for x in range(self.tile_maps[i].width):
                    for y in range(self.tile_maps[i].height):
                        tile = self.tile_maps[i].map[(x, y)]
                        if tile and tile.image_hash:
                            hash_diff = current_image_hash - tile.image_hash
                            if hash_diff < closest_hash_diff:
                                closest_hash_diff = hash_diff
                                closest_position = (x, y)
                                closest_map_index = i

            if closest_position is not None and closest_map_index is not None:
                self.current_tile_map_index = closest_map_index
                self.tile_maps[self.current_tile_map_index].set_position(closest_position)
                print(f"Relocated to tile map {self.current_tile_map_index} at position {closest_position} based on hash similarity.")
            else:
                print("No suitable match found. Keeping current position.")
        else:
            print("Current tile hash is similar. No relocation needed.")


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
        return 0 <= x < tile_map.width and 0 <= y < tile_map.height and tile_map.map[x, y].tile_type != '#' and tile_map.map[x, y].tile_type != 'L'

    
    def press_key(self, key):
        keyboard.press(key)
        time.sleep(0.05)
        keyboard.release(key)

    def press_a(self):
        keyboard.press("a")
        time.sleep(0.05)
        keyboard.release("a")

    def press_random_key(self):
        # Define a list of possible keys to press
        possible_keys = ["up", "left", "right", "down", "a", "b", "a", "b",]
        
        # Randomly select one of the keys
        selected_key = random.choice(possible_keys)
        
        # Press and release the selected key
        keyboard.press(selected_key)
        time.sleep(0.05)
        keyboard.release(selected_key)

        print(f"Pressed key: {selected_key}")  # Optional: Log the pressed key for debugging

    def battle_or_cutscene(self):
        print("Warp not confirmed. Possible battle or cutscene. Spamming")
        self.spam_until_escape()
        self.tile_maps[self.current_tile_map_index].mark_grass()
        self.find_and_relocate_closest_tile()
        self.prev_image = self.capture_screenshot()
        self.tile_maps[self.current_tile_map_index].print_map()

    def warp(self, hit_wall):
        print("Warp confirmed by menu detection.")

        # Handle warp detection
        current_position = self.tile_maps[self.current_tile_map_index].current_position
        if (hit_wall):
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
    
    def check_for_warp_or_battle(self, hit_wall):
        # Check for black screen
        start_time = time.time()
        black_screen = False
        while time.time() - start_time < 3:
            self.current_image = self.capture_screenshot()
            if self.is_black_or_white_screen(self.current_image):
                black_screen = True
            a = not self.in_cutscene()
            if a and black_screen:
                self.warp(hit_wall)
                return
            if a:
                return
        self.battle_or_cutscene()


    def explore(self):
        time.sleep(1)  # Pause for 1 second on startup
        self.prev_image = self.capture_screenshot()

        while True:
            # Use BFS to find the path to the nearest unexplored area
            path = self.bfs_search()
            if not path:
                print("No unexplored Tiles. reseting")
                self.tile_maps = [tilemap.TileMap(800, 800)]  # Start with one tile map
                self.current_tile_map_index = 0  # Index of the current tile map
                self.warp_dict = {}
                path = self.bfs_search()

            for direction, map_index in path:
                # direction = input()
                # image = self.capture_screenshot()

                if self.current_direction != direction:
                    self.turn_direction(direction)
                    if self.in_cutscene():
                        self.check_for_warp_or_battle(True)
                        break

                new_position = self.tile_maps[self.current_tile_map_index].move(direction)
                if (new_position[0] < 0 or new_position[0] >= 800 or new_position[1] < 0 or new_position[1] >= 800):
                    self.tile_maps = [tilemap.TileMap(800, 800)]  # Start with one tile map
                    self.current_tile_map_index = 0  # Index of the current tile map
                    self.warp_dict = {}
                    break

                self.move_direction(direction)

                self.current_image = self.capture_screenshot()

                hit_wall = False
                a = detect_movement(self.prev_image, self.current_image, self.model, count = self.counter)
                if a == 1:
                    self.tile_maps[self.current_tile_map_index].set_position(new_position)
                    self.tile_maps[self.current_tile_map_index].mark_open(new_position, imagehash.average_hash(Image.fromarray(self.current_image)))
                elif a == 0:
                    self.tile_maps[self.current_tile_map_index].mark_wall(new_position)
                    self.press_a()
                    # time.sleep(.2)
                    hit_wall = True
                    # self.handle_trapped_scenario()
                elif a == 2:
                    self.tile_maps[self.current_tile_map_index].set_position(new_position)
                    self.tile_maps[self.current_tile_map_index].mark_ledge(new_position)
                    new_position = self.tile_maps[self.current_tile_map_index].move(direction)
                    self.tile_maps[self.current_tile_map_index].set_position(new_position)
                self.counter += 1

                self.prev_image = self.current_image
                # if random.random() > .5:
                #     self.press_a()
                #     time.sleep(.2)
                # else:
                #     self.press_key("b")
                
                self.tile_maps[self.current_tile_map_index].print_map()
                if self.in_cutscene():
                    self.check_for_warp_or_battle(hit_wall)
                    break


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
model = tf.keras.models.load_model('movement_detection_model.h5')
explorer = PokemonExplorer(bbox=bbox, model=model)
explorer.explore()
