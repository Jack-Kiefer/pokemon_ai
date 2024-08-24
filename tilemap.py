import numpy as np
import imagehash

class Tile:
    def __init__(self, tile_type=' ', image_hash=None):
        self.tile_type = tile_type  # Tile type (' ', '#', '.', etc.)
        self.image_hash = image_hash  # Perceptual hash of the tile image

class TileMap:
    def __init__(self, width, height):
        # Initialize a map of Tile objects
        self.map = np.full((width, height), None)  
        for x in range(width):
            for y in range(height):
                self.map[x, y] = Tile()  # Initialize each position with an empty Tile
        self.width = width
        self.height = height
        self.current_position = (width // 2, height // 2)  # Start at the center
        self.min_x, self.min_y = self.current_position  # Bounds of explored area
        self.max_x, self.max_y = self.current_position

        # Mark the starting position as explored
        self.map[self.current_position].tile_type = '.'

    def check_hash(self, hash):
        if (self.map[self.current_position].image_hash == None):
            return True
        else:
            if (abs(hash - self.map[self.current_position].image_hash) < 3):
                print(abs(hash - self.map[self.current_position].image_hash))
                return True
            else:
                return False


    def reset(self):
        for x in range(self.width):
            for y in range(self.height):
                self.map[x, y] = Tile()  # Reset each position with a new empty Tile
        self.current_position = (self.width // 2, self.height // 2)  # Reset to the center
        self.min_x, self.min_y = self.current_position  # Reset bounds
        self.max_x, self.max_y = self.current_position
        self.map[self.current_position].tile_type = '.'  # Mark starting position as explored

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
        if self.map[position].tile_type == '.':
            self.map[position].tile_type = 'L'
        else:
            self.map[position].tile_type = '#'
        self.update_bounds(position)

    def mark_grass(self):
        self.map[self.current_position].tile_type = 'G'
        

    def mark_open(self, position, image_hash):
        tile = self.map[position]
        tile.tile_type = '.'
        if image_hash == None:
            tile.image_hash = image_hash
        self.update_bounds(position)

    def set_position(self, position, image_hash=None):
        self.current_position = position
        tile = self.map[position]
        if tile.tile_type != 'W':
            tile.tile_type = '.'
        if image_hash is not None:
            tile.image_hash = image_hash
        self.update_bounds(position)

    def update_bounds(self, position):
        x, y = position
        self.min_x = min(self.min_x, x)
        self.max_x = max(self.max_x, x)
        self.min_y = min(self.min_y, y)
        self.max_x = max(self.max_x, x)
        self.max_y = max(self.max_y, y)

    def print_map(self):
        # Get the current position
        x, y = self.current_position

        # Define the boundaries of the 40x40 area around the player
        min_x = max(x - 20, 0)
        max_x = min(x + 20, self.width - 1)
        min_y = max(y - 20, 0)
        max_y = min(y + 20, self.height - 1)

        # Temporarily mark the current position as 'X' for printing
        temp_map = np.full((self.width, self.height), ' ')
        for i in range(min_x, max_x + 1):
            for j in range(min_y, max_y + 1):
                temp_map[i, j] = self.map[i, j].tile_type

        temp_map[x, y] = 'X'

        print("\nCurrent Map (40x40 view):")
        for j in range(min_y, max_y + 1):
            row = "".join(temp_map[i][j] for i in range(min_x, max_x + 1))
            print(row)
        print()
