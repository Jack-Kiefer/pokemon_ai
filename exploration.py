import numpy as np
from PIL import ImageGrab, Image
import time
import pydirectinput
import imagehash
import random
import keyboard

class TileMap:
    def __init__(self, width, height):
        self.map = np.full((width, height), ' ')  # Initialize an empty map with ' ' as unexplored
        self.width = width
        self.height = height
        self.current_position = (width // 2, height // 2)  # Start at the center
        self.current_direction = 'left'  # Start facing 'left'
        self.min_x, self.min_y = self.current_position  # Bounds of explored area
        self.max_x, self.max_y = self.current_position

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

class PokemonExplorer:
    def __init__(self, bbox, interval=0.4, pixel_size=(32, 32), hash_size=8, hash_threshold=5):
        self.bbox = bbox
        self.interval = interval
        self.pixel_size = pixel_size
        self.hash_size = hash_size
        self.hash_threshold = hash_threshold
        self.tile_map = TileMap(100, 100)  # Create a 100x100 map
        self.last_hash = None

    def perceptual_hash(self, image):
        return imagehash.average_hash(image, hash_size=self.hash_size)

    def capture_screenshot(self):
        screenshot = ImageGrab.grab(bbox=self.bbox)
        return screenshot.resize(self.pixel_size, Image.BILINEAR)

    def perform_action(self, action):
        actions = {'up': 'up', 'down': 'down', 'left': 'left', 'right': 'right'}
        pydirectinput.press(actions[action], interval=.1)

    def turn_direction(self, direction, min_hold_time=0.05, wait_time=.5):
        keyboard.press(direction)   # Press and hold the key down
        time.sleep(min_hold_time)   # Hold it for the minimum time
        keyboard.release(direction) # Release the key
        time.sleep(wait_time)   # Small delay before the next key press


    def explore(self):
        directions = ['up', 'down', 'left', 'right']
        
        while True:  # Loop indefinitely to continue exploring
            direction = random.choice(directions)  # Choose a random direction
            if self.tile_map.current_direction != direction:
                # If the direction is different, first turn towards it
                self.turn_direction(direction)
                self.tile_map.set_direction(direction)

            # Compute the new position
            new_position = self.tile_map.move(direction)

            # Perform the movement in the desired direction
            self.turn_direction(direction)

            # Capture the screenshot and compute the hash
            new_screenshot = self.capture_screenshot()
            new_hash = self.perceptual_hash(new_screenshot)

            if self.last_hash is None:
                # First move, initialize last_hash and set position
                self.last_hash = new_hash
                self.tile_map.set_position(new_position)
                self.tile_map.mark_open(new_position)
            elif abs(new_hash - self.last_hash) > 2:
                # If the hash changed, the move was successful
                self.tile_map.set_position(new_position)
                self.tile_map.mark_open(new_position)
                self.last_hash = new_hash
            else:
                # If the hash did not change, mark the position as a wall
                self.tile_map.mark_wall(new_position)
            self.tile_map.print_map()

# Example usage
bbox = (1250, 110, 2600, 1100)  # Coordinates for the region you want to capture
explorer = PokemonExplorer(bbox=bbox)
explorer.explore()
