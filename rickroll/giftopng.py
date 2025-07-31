import cv2
import numpy as np
import os

spec_dir = 'rickroll/frames_pngs/'

# define output directories for the PNG images and coordinates
output_dir = 'rickroll/outline_pngs/'
coords_dir = 'rickroll/frame_coords/'

# make sure the directories exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(coords_dir, exist_ok=True)

for gifpart in os.listdir(spec_dir):

    # load the black-and-white image in grayscale
    image = cv2.imread(spec_dir + gifpart, cv2.IMREAD_GRAYSCALE)

    # Define the range for light gray and black color (we want to keep these regions)
    black_threshold = 50  # Black pixel intensity threshold (0-255)
    light_gray_min = 100  # Minimum intensity for light gray
    light_gray_max = 220  # Maximum intensity for light gray

    # create a mask identifying the black and light gray areas
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[(image <= black_threshold) | ((image >= light_gray_min) & (image <= light_gray_max))] = 255

    # get the coordinates of the black and light gray areas
    coordinates = np.column_stack(np.where(mask == 255))

    # define the number of dots
    N = 600  # Total number of dots

    # evenly sample the coordinates from the black and light gray areas
    # we want to distribute N points over the identified area
    step_size = max(1, len(coordinates) // N)

    # select the coordinates for the dots (evenly spaced)
    selected_coords = coordinates[::step_size][:N]

    # create a blank white image to draw the dots
    output_image = np.ones_like(image, dtype=np.uint8) * 255  # White background

    # draw the dots (black color) on the image
    for (y, x) in selected_coords:
        cv2.circle(output_image, (x, y), radius=1, color=(0, 0, 0), thickness=-1)  # Black dot

    # save the image with dots as a .png file
    cv2.imwrite(output_dir + gifpart.replace('.gif', '.png'), output_image)

    # save the coordinates of the dots as a .csv file
    coords_file = os.path.join(coords_dir, gifpart.replace('.png', '_coords.csv'))
    np.savetxt(coords_file, selected_coords, delimiter=',')

    print(f"Processed and saved: {gifpart.replace('.gif', '.png')} and coordinates to {coords_file}")



from PIL import Image
import os
import re

# set the output directory containing the PNG files
output_dir = 'rickroll/'

# function to extract the numeric part from the filename for sorting
def extract_number(filename):
    match = re.search(r'(\d+)', filename)  # Find the first number in the filename
    return int(match.group(1)) if match else 0

# get all the PNG files in the directory (sorted by the numeric part of the filename)
images = [Image.open(os.path.join(output_dir, filename)) for filename in sorted(os.listdir(output_dir), key=extract_number) if filename.endswith('.png')]

# define the output gif file path
gif_output_path = 'output_rr.gif'

# save the images as a gif at 24 frames per second
images[0].save(gif_output_path, save_all=True, append_images=images[1:], optimize=False, duration=1000//12, loop=0)

print(f"GIF saved as {gif_output_path}")

