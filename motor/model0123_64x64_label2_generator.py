import os
import tensorflow as tf
from tensorflow import keras
from model0123_64x64_label2 import create_motor_generator
import numpy as np

# 設定パラメータ
latent_dim = 100
num_classes = 2
num_img_per_label = 1

# モデルのパス
model_path = 'motor/output/model0123_64x64_label2'
generator_paths = [os.path.join(model_path, d) for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]

# 生成器のロード
generator = create_motor_generator(latent_dim, num_classes)

for folder_path in generator_paths:
    generator_weights_path = os.path.join(folder_path, 'generator.weights.h5')
    try:
        generator.load_weights(generator_weights_path)
        print(f"Successfully loaded weights from {generator_weights_path}")

        # 各ラベルに対して画像を生成
        for label1 in [40, 80, 120, 160, 200, 240]:
            for label2 in [40, 80, 120, 160, 200, 240]:
                # 当該ラベルの繰り返し
                labels = tf.constant([label1, label2] * num_img_per_label, dtype=np.float32)
                labels = tf.reshape(labels, (num_img_per_label, 2))  # ラベルの形状を調整

                # 潜在空間からのランダムベクトル生成
                random_latent_vectors = tf.random.normal(shape=(num_img_per_label, latent_dim))
                generator_input = tf.concat([random_latent_vectors, labels], axis=1)

                # 条件付きで偽画像を生成
                generated_images = generator(generator_input, training=False)
                generated_images = (generated_images * 127.5) + 127.5  # [-1, 1]から[0, 255]へスケール変換
                generated_images = tf.cast(generated_images, tf.uint8)  # 整数型へ変換

                # 画像を保存
                for i in range(num_img_per_label):
                    img = keras.preprocessing.image.array_to_img(generated_images[i])
                    save_path = os.path.join(folder_path, "generated_img_label_%d_%d_%d.png" % (label1, label2, i))
                    img.save(save_path)
    except Exception as e:
        print(f"Error loading weights from {generator_weights_path}: {e}")

print("画像生成が完了しました。")
