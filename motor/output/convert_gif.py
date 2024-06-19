import os
import glob
from PIL import Image

def create_gif_for_all_subdirs(base_dir, output_dir):
    # 出力ディレクトリが存在しない場合は作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 基本ディレクトリの中にある4桁のディレクトリを取得
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and len(d) == 4 and d.isdigit()]

    for subdir in subdirs:
        dir_path = os.path.join(base_dir, subdir)
        # 画像ファイルのパスパターンを指定
        file_pattern = os.path.join(dir_path, "generated_img_label_*_0.png")
        # 画像ファイルを読み込む
        images = []
        for filename in sorted(glob.glob(file_pattern)):
            img = Image.open(filename)
            images.append(img)

        if images:
            # GIFファイルの出力パスを指定
            gif_path = os.path.join(output_dir, f"{subdir}_output.gif")
            # GIFを作成して保存
            images[0].save(gif_path, save_all=True, append_images=images[1:], duration=500, loop=0)
            print(f"Created GIF: {len(images)}")
        else:
            print(f"No images found in {dir_path}")

# 使用例
base_dir = "motor/output/sample"
output_dir = "motor/output/sample/gif"
create_gif_for_all_subdirs(base_dir, output_dir)
