from tensorflow import keras
import matplotlib.pyplot as plt
import os
import gdown
from zipfile import ZipFile


# ハイパーパラメータ
BATCH_SIZE = 32

if __name__ == "__main__":
    # データセット用意
    dataset_dir = "celeba_gan"

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    url = "https://drive.google.com/uc?id=1O7m1010EJjLE5QxLZiM9Fpjs7Oj6e684"
    output = "celeba_gan/data.zip"

    if not os.path.exists(output):
        gdown.download(url, output, quiet=True)

        with ZipFile("celeba_gan/data.zip", "r") as z:
            z.extractall("celeba_gan")

    dataset = keras.utils.image_dataset_from_directory(
        "celeba_gan", label_mode=None, image_size=(64, 64), batch_size=BATCH_SIZE
    )
    dataset = dataset.map(lambda x: x / 255.0)

    for x in dataset:
        plt.axis("off")
        plt.imshow((x.numpy() * 255).astype("int32")[0])
        break
