import os
from matplotlib import pyplot as plt
import numpy as np
import cv2
import random

# Paths and directories
OUTPUT_DIR = 'cGAN_practice/datasets/images'

# Image parameters
img_qty = 9
height, width = 128, 128

# Get all image paths
all_images = [os.path.join(OUTPUT_DIR, img) for img in os.listdir(OUTPUT_DIR) if img.endswith('.png')]

# Randomly select 9 images
selected_images = random.sample(all_images, img_qty)

# Create a 3x3 grid of images
fig, axes = plt.subplots(3, 3, figsize=(10, 10))

for i, img_path in enumerate(selected_images):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    ax = axes[i//3, i%3]
    ax.imshow(img, cmap='gray')
    ax.axis('off')

plt.tight_layout()
plt.show()
