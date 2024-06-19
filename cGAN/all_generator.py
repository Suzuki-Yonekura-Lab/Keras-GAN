import os
import cv2
import numpy as np
from scipy.ndimage import label
import matplotlib.pyplot as plt

def binarize_image(image_path):
    """
    画像をグレースケールで読み込んでから二値化する。
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    return binary_image

def count_connected_components(binary_image):
    """
    二値化画像から連結成分の数をカウントする。
    """
    structure = np.ones((3, 3), dtype=int)  # 8-connectivity
    _, n_components = label(binary_image, structure)
    return n_components

def analyze_images(base_directory):
    """
    指定されたディレクトリの画像を解析し、連結成分数を集計する。
    """
    # 結果を格納する辞書
    results = {0: [0, 0, 0, 0], 1: [0, 0, 0, 0], 2: [0, 0, 0, 0]}

    # 画像ファイルのパスを生成
    for label in [0, 1, 2]:
        for z in range(100):  # zは0から99まで
            img_path = os.path.join(base_directory, f'generated_img_label_{label}_{z}.png')
            if os.path.exists(img_path):
                binary_image = binarize_image(img_path)
                n_components = count_connected_components(binary_image)

                # 連結成分数に基づいてカウントを更新
                if n_components == 1:
                    results[label][0] += 1
                elif n_components == 2:
                    results[label][1] += 1
                elif n_components == 3:
                    results[label][2] += 1
                else:
                    results[label][3] += 1

    return results

# 画像の基本ディレクトリを設定
base_directory = 'cGAN/generated_images_samplev6'

# 画像を解析
results = analyze_images(base_directory)

# 結果を出力
print("連結成分数の集計結果:")
for label in results:
    print(f"Label {label}: 1 component - {results[label][0]}, 2 components - {results[label][1]}, 3 components - {results[label][2]}, others - {results[label][3]}")
