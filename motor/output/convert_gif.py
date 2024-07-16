import os
import glob
from PIL import Image

def create_gif_for_all_subdirs(base_dir, output_dir):
    # 出力ディレクトリが存在しない場合は作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 基本ディレクトリの中にあるサブディレクトリを取得
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    for subdir in subdirs:
        # サブディレクトリの名前を取得
        subdir_name = os.path.basename(subdir)

        # label1固定でlabel2が変わるGIFを生成
        for label1 in [40, 80, 120, 160, 200, 240]:
            file_pattern = os.path.join(subdir, f"generated_img_label_{label1}_*_0.png")
            images = []
            for filename in sorted(glob.glob(file_pattern)):
                img = Image.open(filename)
                images.append(img)

            if images:
                gif_path = os.path.join(output_dir, f"{subdir_name}_label1_{label1}_varying_label2.gif")
                images[0].save(gif_path, save_all=True, append_images=images[1:], duration=500, loop=0)
                print(f"Created GIF: {len(images)} images in {subdir} for label1={label1}")
            else:
                print(f"No images found in {subdir} for label1={label1}")

        # label2固定でlabel1が変わるGIFを生成
        for label2 in [40, 80, 120, 160, 200, 240]:
            file_pattern = os.path.join(subdir, f"generated_img_label_*_{label2}_0.png")
            images = []
            for filename in sorted(glob.glob(file_pattern)):
                img = Image.open(filename)
                images.append(img)

            if images:
                gif_path = os.path.join(output_dir, f"{subdir_name}_label2_{label2}_varying_label1.gif")
                images[0].save(gif_path, save_all=True, append_images=images[1:], duration=500, loop=0)
                print(f"Created GIF: {len(images)} images in {subdir} for label2={label2}")
            else:
                print(f"No images found in {subdir} for label2={label2}")

# 使用例
base_dir = "motor/output/mochu"
output_dir = "motor/output/mochu_gif"
create_gif_for_all_subdirs(base_dir, output_dir)
