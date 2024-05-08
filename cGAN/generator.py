import os
import tensorflow as tf
from tensorflow import keras
from samplev3 import create_cgan_generator


# 設定パラメータ
latent_dim = 128
num_classes = 3
num_images_per_label = 10

# モデルのパス
model_path = 'cGAN/output/samplev3-batch128-epoch50-lambda20-latent128-data6144'
generator_path = os.path.join(model_path, 'generator_weights.h5')

# 生成器のロード
generator = create_cgan_generator(latent_dim, num_classes)
generator.load_weights(generator_path)

# 画像生成と保存のためのディレクトリ
output_dir = 'cGAN/generated_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 各ラベルに対して画像を生成
for label in range(num_classes):
    # 当該ラベルのためのランダムベクトル
    random_latent_vectors = tf.random.normal(shape=(num_images_per_label, latent_dim))
    labels = tf.constant([label] * num_images_per_label)
    labels = tf.reshape(labels, (num_images_per_label, 1))

    # 条件付きで偽画像を生成
    generated_images = generator([random_latent_vectors, labels], training=False)
    generated_images = (generated_images * 127.5) + 127.5  # [-1, 1]から[0, 255]へスケール変換
    generated_images = tf.cast(generated_images, tf.uint8)  # 整数型へ変換

    # 生成された画像を保存
    for i, img_tensor in enumerate(generated_images):
        img = keras.preprocessing.image.array_to_img(img_tensor)
        img.save(os.path.join(output_dir, f'generated_img_label_{label}_{i}.png'))

print("画像生成が完了しました。")
