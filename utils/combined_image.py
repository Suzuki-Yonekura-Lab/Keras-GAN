from PIL import Image

# 9つの画像のファイル名をリストに格納する
image_files = ["GAN/output/generated_img_{:03d}_{}.png".format(46, n) for n in range(1, 10)]

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
new_image.save("utils/output/combined_image.png")
