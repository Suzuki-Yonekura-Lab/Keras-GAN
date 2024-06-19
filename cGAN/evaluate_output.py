import os
import cv2
import numpy as np
from scipy.ndimage import label
import matplotlib.pyplot as plt

def binarize_image(image):
    """
    画像をグレースケールで読み込んでから二値化する。
    """
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

def analyze_images(base_dir, step=10, max_x=2080, labels=[0, 1, 2]):
    """
    画像を解析し、各 x の値で正確に予測された連結成分の数をカウントする。
    """
    counts = {x: 0 for x in range(0, max_x + 1, step)}
    for x in range(0, max_x + 1, step):
        x_formatted = str(x).zfill(3)  # x の値を3桁の文字列にフォーマット
        for y in labels:
            for z in range(3):  # 各組み合わせに対して3枚の画像がある
                path = f"{base_dir}/generated_img_{x_formatted}_label_{y}_{z}.png"
                image = cv2.imread(path)
                if image is not None:
                    binary_image = binarize_image(image)
                    components = count_connected_components(binary_image)
                    if components == y + 1:
                        counts[x] += 1
    return counts

def find_x_with_match_count(results, match_count=6):
    """
    一致数が特定の値であるxの値を見つける。
    """
    matching_xs = [x for x, count in results.items() if count == match_count]
    return matching_xs

# 画像の基本ディレクトリを設定
base_directory = 'cGAN/output/samplev6'

# 画像を解析
results = analyze_images(base_directory)
print(find_x_with_match_count(results, 6))

# 結果をプロット
plt.figure(figsize=(10, 5))
plt.plot(list(results.keys()), list(results.values()), marker='o')
plt.title('Match Count Across Different Epoch')
plt.xlabel('Epoch')
plt.ylabel('Match Count')
plt.grid(True)
plt.show()
