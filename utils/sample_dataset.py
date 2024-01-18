import os
import glob
import cv2
import matplotlib.pyplot as plt


# 自作データセットのパスを指定
dataset_dir = 'WGAN-gp/datasets'
categories = ['rectangle', 'circle', 'doughnut']

num_samples_per_category = 5  # 各カテゴリから表示するサンプルの数

plt.figure(figsize=(15, 5))

for idx, category in enumerate(categories):
    # カテゴリ内の画像ファイルのパスを取得
    image_files = glob.glob(os.path.join(dataset_dir, category, '*.png'))
    image_files = sorted(image_files)[:num_samples_per_category]  # 最初のいくつかをサンプリング

    for i, image_file in enumerate(image_files):
        img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)  # 画像を読み込み

        # 画像を表示
        ax = plt.subplot(len(categories), num_samples_per_category, idx * num_samples_per_category + i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"{category}")
        plt.axis('off')

plt.tight_layout()
plt.show()
