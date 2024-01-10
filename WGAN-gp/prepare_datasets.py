import os
import numpy as np
import cv2

OUTPUT_DIR = 'WGAN-gp/datasets'
rng = np.random.default_rng(0)
height = 128
width = 128

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
if not os.path.exists(OUTPUT_DIR + "/rectangle"):
    os.makedirs(OUTPUT_DIR + "/rectangle")
if not os.path.exists(OUTPUT_DIR + "/circle"):
    os.makedirs(OUTPUT_DIR + "/circle")
if not os.path.exists(OUTPUT_DIR + "/doughnut"):
    os.makedirs(OUTPUT_DIR + "/doughnut")

img_qty = 256  # quantity of images for each shape

# rectangle
for img_id in range(img_qty):
    x_1, y_1, x_2, y_2 = np.random.randint(1, height, 4)
    img = np.zeros((height, width))  # initialize
    img = cv2.rectangle(img, (x_1, y_1), (x_2, y_2), color=(255, 255, 255), thickness=-1)
    cv2.imwrite(f'{OUTPUT_DIR}/rectangle/{img_id}.png', img)

# circle
min_radius = 10
for img_id in range(img_qty):
    x_center, y_center = np.random.randint(min_radius+1, height-min_radius-1, 2)
    distance_to_border = np.min([x_center, width-x_center, y_center, height-y_center])
    radius = np.random.randint(min_radius, distance_to_border)
    img = np.zeros((height, width))  # initialize
    img = cv2.circle(img, (x_center, y_center), radius, (255, 255, 255), thickness=-1)  # circle
    cv2.imwrite(f'{OUTPUT_DIR}/circle/{img_id}.png', img)

# doughnut
min_hole_radius = 8
min_diff = 2
min_radius  = min_hole_radius + min_diff
for img_id in range(img_qty):
    x_center, y_center = np.random.randint(min_radius+1, height-min_radius-1, 2)
    distance_to_border = np.min([x_center, width-x_center, y_center, height-y_center])
    radius = np.random.randint(min_radius, distance_to_border)
    thickness = np.random.randint(min_hole_radius, radius)
    img = np.zeros((height, width))  # initialize
    img = cv2.circle(img, (x_center, y_center), radius, (255, 255, 255), thickness=thickness)  # donut
    cv2.imwrite(f'{OUTPUT_DIR}/doughnut/{img_id}.png', img)
