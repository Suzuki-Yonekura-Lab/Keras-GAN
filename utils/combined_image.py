import os
from PIL import Image, ImageDraw, ImageFont


def combine(num, output_name):
    # 9つの画像のファイル名をリストに格納する
    image_files = ["WGAN-gp/output/" + output_name + "/generated_img_{:03d}_{}.png".format(num-1, n) for n in range(1, 10)]

    # 9つの画像を読み込む
    images = [Image.open(filename) for filename in image_files]

    # 3x3の新しい画像を作成する
    new_image = Image.new('RGB', (images[0].width * 3, images[0].height * 3))

    # 9つの画像を3x3の配置に結合する
    for index, image in enumerate(images):
        row = index // 3
        col = index % 3
        new_image.paste(image, (image.width * col, image.height * row))

    # 新しい画像を保存する
    output_dir = "utils/output/" + output_name

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    new_image.save(output_dir + "/combined_image" + str(num) + ".png")


def combine_images(nums, output_name):
    # 各画像のファイル名をリストに格納する
    image_files = [
        "utils/output/" + output_name + f"/combined_image{num}.png" for num in nums
    ]

    # 画像を読み込む
    images = [Image.open(filename) for filename in image_files]

    # 画像間のスペーシングを設定
    spacing = 10

    # タイトル用のスペースを設定
    title_space = 30

    # 縦2行、横3列のレイアウトで全体の新しい画像を作成する
    total_width = sum(image.width for image in images[:3]) + spacing * 4
    total_height = sum(image.height for image in images[:2]) + spacing * 3 + title_space * 2
    final_image = Image.new('RGB', (total_width, total_height), "white")

    font = ImageFont.load_default().font_variant(size=24)

    # 画像とタイトルを縦2 x 横3の形で結合する
    y_offset = spacing
    for i in range(2):
        x_offset = spacing
        for j in range(3):
            index = i * 3 + j
            image = images[index]

            # 画像をペースト
            final_image.paste(image, (x_offset, y_offset))

            # タイトル（エポック番号）を追加
            draw = ImageDraw.Draw(final_image)
            draw.text((x_offset, y_offset + image.height), f"Epoch {nums[index]}", fill="black", font=font)

            x_offset += image.width + spacing
        y_offset += images[i * 3].height + spacing + title_space

    # 最終的な画像を保存するディレクトリを設定
    output_dir = "utils/output/" + output_name

    # ディレクトリが存在しない場合は作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 画像を保存する
    final_image_path = output_dir + "/combined_vertical_horizontal_with_titles.png"
    final_image.save(final_image_path)


# 6つの異なる組み合わせ
nums = [1, 10, 20, 30, 40, 50]
output_name = "gan-batch128-epoch50-latent128-data3072"

for num in nums:
    combine(num, output_name)

# 関数を実行して結合された画像を生成する
combine_images(nums, output_name)
