import os
from PIL import ImageGrab, Image

def capture_and_save_tiles(bbox, grid_size=(16, 16), tile_size=(32, 32), output_dir='tiles_output'):
    """
    Capture a screenshot of the specified bounding box, split it into a grid of tiles,
    and save each tile as an image in the specified output directory.

    :param bbox: A tuple (x1, y1, x2, y2) defining the region to capture.
    :param grid_size: A tuple (rows, cols) specifying the number of tiles in the grid.
    :param tile_size: A tuple (width, height) specifying the size of each tile.
    :param output_dir: The directory to save the tile images.
    """
    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Capture the screenshot
    screenshot = ImageGrab.grab(bbox=bbox)
    
    # Resize the screenshot to match the grid size and tile size
    screenshot = screenshot.resize((grid_size[1] * tile_size[0], grid_size[0] * tile_size[1]), Image.BILINEAR)
    
    # Get the dimensions of the resized screenshot
    width, height = screenshot.size
    
    # Calculate the width and height of each tile
    tile_width = width // grid_size[1]
    tile_height = height // grid_size[0]

    # Split the screenshot into tiles and save each tile
    for i in range(grid_size[0]):  # rows
        for j in range(grid_size[1]):  # cols
            left = j * tile_width
            top = i * tile_height
            right = left + tile_width
            bottom = top + tile_height

            # Crop the tile from the screenshot
            tile = screenshot.crop((left, top, right, bottom))

            # Save the tile as an image
            tile_filename = os.path.join(output_dir, f'tile_{i}_{j}.png')
            tile.save(tile_filename)
            print(f"Saved tile: {tile_filename}")

# Example usage
bbox = (1275, 135, 2565, 1100)  # Coordinates for the region to capture
capture_and_save_tiles(bbox, grid_size=(16, 16), tile_size=(32, 32), output_dir='tiles_output')
