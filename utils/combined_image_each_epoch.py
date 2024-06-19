import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# xの値を指定
x_value = 1770

# 基本ディレクトリとサブディレクトリのリスト
base_directory = 'cGAN/output/samplev5'
labels = [0, 1, 2]  # 期待される連結成分数 - 1
repeats = [0, 1, 2]  # 同じ条件で生成された画像のインデックス

# 各ディレクトリから画像を読み込む
image_files = [os.path.join(base_directory, f"generated_img_{x_value}_label_{y}_{z}.png") for z in repeats for y in labels]

# 画像を表示するための3x3グリッドを作成
fig, axes = plt.subplots(3, 3, figsize=(10, 10))

# タイトルを設定
titles = ["1 circle", "2 circles", "3 circles"]
for i, title in enumerate(titles):
    axes[0, i].set_title(title, fontsize=20)

# 画像をグリッドに配置
for i, img_path in enumerate(image_files):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # グレースケールで画像を読み込む
    ax = axes[i // 3, i % 3]
    if img is not None:
        ax.imshow(img, cmap='gray')  # グレースケール画像として表示
    ax.axis('off')  # 軸を非表示にする

    # 赤い四角を追加する必要がある位置を指定
    if (i // 3 == 0 and i % 3 == 0) or (i // 3 == 1 and i % 3 == 0) or \
            (i // 3 == 0 and i % 3 == 1) or (i // 3 == 2 and i % 3 == 1) or \
            (i // 3 == 0 and i % 3 == 2) or (i // 3 == 2 and i % 3 == 2):
        # 赤い枠を追加
        rect = Rectangle((0, 0), 1, 1, fill=False, edgecolor='red', linewidth=20, transform=ax.transAxes)
        ax.add_patch(rect)

plt.tight_layout()  # レイアウトの調整
plt.show()  # 画像の表示
