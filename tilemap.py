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
    def __init__(self, width, height, direction = "left"):
        self.map = np.full((width, height), ' ')  # Initialize an empty map with ' ' as unexplored
        self.width = width
        self.height = height
        self.current_position = (width // 2, height // 2)  # Start at the center
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
        if (self.map[position] != 'W'):
            self.map[position] = '.'
            self.update_bounds(position)

    def set_position(self, position):
        self.current_position = position
        if (self.map[position] != 'W'):
            self.map[position] = '.'  # Mark the AI's current position as open space
        self.update_bounds(position)

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