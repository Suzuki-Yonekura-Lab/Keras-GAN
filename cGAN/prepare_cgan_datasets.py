import os
import numpy as np
import cv2

def check_non_overlap(centers, radiuses, new_center, new_radius):
    """ Check if a new circle overlaps with any existing circles. """
    for center, radius in zip(centers, radiuses):
        dist = np.sqrt((center[0] - new_center[0])**2 + (center[1] - new_center[1])**2)
        if dist < radius + new_radius:
            return False
    return True

def draw_circle(img, centers, radiuses):
    """ Draw circles on the image. """
    for (x_center, y_center), radius in zip(centers, radiuses):
        cv2.circle(img, (x_center, y_center), radius, (255, 255, 255), thickness=-1)

# Paths and directories
OUTPUT_DIR = 'WGAN-gp/datasets/cgan'
sub_dirs = ["circle1", "circle2", "circle3"]
for sub_dir in sub_dirs:
    path = os.path.join(OUTPUT_DIR, sub_dir)
    if not os.path.exists(path):
        os.makedirs(path)

# Image parameters
img_qty = 512
height, width = 128, 128
min_radius = 20

# Generate images with circles
rng = np.random.default_rng(0)
for img_id in range(img_qty):
    for num_circles in [1, 2, 3]:
        img = np.zeros((height, width))
        centers = []
        radiuses = []
        attempts = 0
        max_attempts = 1000  # Maximum attempts to place circles

        while len(centers) < num_circles and attempts < max_attempts:
            x_center, y_center = rng.integers(min_radius + 1, height - min_radius - 1, 2)
            distance_to_border = np.min([x_center, width - x_center, y_center, height - y_center])
            radius = rng.integers(min_radius, distance_to_border)

            if check_non_overlap(centers, radiuses, (x_center, y_center), radius):
                centers.append((x_center, y_center))
                radiuses.append(radius)
            attempts += 1

        if len(centers) == num_circles:
            draw_circle(img, centers, radiuses)
            cv2.imwrite(f'{OUTPUT_DIR}/circle{num_circles}/{img_id}.png', img)
        else:
            print(f"Failed to place {num_circles} circles for image {img_id}. Trying again.")
            img_id -= 1  # Decrement to retry the image generation
