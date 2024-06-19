import os
import tensorflow as tf
from tensorflow import keras
from samplev5 import create_cgan_generator


# 設定パラメータ
latent_dim = 100
num_classes = 3
num_images_per_label = 100

# モデルのパス
model_path = 'cGAN/output/samplev6'
generator_path = os.path.join(model_path, 'generator_1580.weights.h5')

# 生成器のロード
generator = create_cgan_generator(latent_dim, num_classes)
generator.load_weights(generator_path)
# 画像生成と保存のためのディレクトリ
output_dir = 'cGAN/generated_images_samplev6'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 各ラベルに対して画像を生成
for label in range(num_classes):
    # 当該ラベルの繰り返し
    labels = tf.constant([label] * num_images_per_label)
    labels = tf.reshape(labels, (num_images_per_label, 1))  # ラベルの形状を調整

    # 潜在空間からのランダムベクトル生成
    random_latent_vectors = tf.random.normal(shape=(num_images_per_label, latent_dim))
    labels_one_hot = tf.one_hot(labels, depth=3)
    labels_one_hot = tf.reshape(labels_one_hot, shape=(num_images_per_label, 3))
    generator_input = tf.concat([random_latent_vectors, labels_one_hot], axis=1)

    # 条件付きで偽画像を生成
    generated_images = generator(generator_input, training=False)
    generated_images = (generated_images * 127.5) + 127.5  # [-1, 1]から[0, 255]へスケール変換
    generated_images = tf.cast(generated_images, tf.uint8)  # 整数型へ変換

    # 生成された画像を保存
    for i, img_tensor in enumerate(generated_images):
        img = keras.preprocessing.image.array_to_img(img_tensor)
        img.save(os.path.join(output_dir, f'generated_img_label_{label}_{i}.png'))

print("画像生成が完了しました。")
