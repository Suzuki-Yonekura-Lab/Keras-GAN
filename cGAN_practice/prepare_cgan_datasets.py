import os
import numpy as np
import cv2
import pandas as pd

def draw_circle(img, center, radius):
    """ Draw a circle on the image. """
    cv2.circle(img, center, radius, (255, 255, 255), thickness=-1)

# Paths and directories
OUTPUT_DIR = 'cGAN_practice/datasets/images'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Image parameters
img_qty = 2048
height, width = 128, 128
radiuses = []
min_radius = 20

# Generate images with circles
rng = np.random.default_rng(0)
for img_id in range(img_qty):
    img = np.zeros((height, width))
    center = (height // 2, width // 2)
    radius = rng.integers(min_radius, height // 2)
    radiuses.append(radius)

    draw_circle(img, center, radius)
    cv2.imwrite(f'{OUTPUT_DIR}/{img_id}.png', img)

# Save radiuses to csv
radiuses_df = pd.DataFrame(radiuses, columns=['radius'])
radiuses_df.to_csv('cGAN_practice/datasets/radiuses.csv', index=False)
