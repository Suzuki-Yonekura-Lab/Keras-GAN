import os
import cv2
import matplotlib.pyplot as plt

# ディレクトリの設定
directories = ['cGAN/archive-datasets/circle1', 'cGAN/archive-datasets/circle2', 'cGAN/archive-datasets/circle3']
titles = ["1 circle", "2 circles", "3 circles"]  # 各カラムのタイトル

# 各ディレクトリから画像を読み込む
image_files = [os.path.join(dir, f"{i}.png") for dir in directories for i in range(3)]

# 画像を表示するための3x3グリッドを作成
fig, axes = plt.subplots(3, 3, figsize=(10, 15))  # サイズを調整して見やすくする

for col, title in enumerate(titles):
    axes[0, col].set_title(title, fontsize=20)  # 各カラムのタイトルを設定

for i, img_path in enumerate(image_files):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # グレースケールで画像を読み込む
    ax = axes[i % 3, i // 3]  # カラムごとに縦に並べる
    ax.imshow(img, cmap='gray')  # グレースケール画像として表示
    ax.axis('off')  # 軸を非表示にする

plt.tight_layout()  # レイアウトの調整
plt.show()  # 画像の表示
