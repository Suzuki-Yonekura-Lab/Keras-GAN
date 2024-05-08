import glob
import csv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Reshape
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt


# ハイパーパラメータ
BATCH_SIZE = 128
LATENT_DIM = 128
EPOCHS = 50
LAMBDA_GP = 20


def load_dataset(dir, image_size=(128, 128)):
    images = []
    labels = []  # Store the conditions here
    for filename in sorted(glob.glob(dir + '/images/*.png')):
        img = load_img(filename, target_size=image_size, color_mode='grayscale')
        img = img_to_array(img)
        img = (img - 127.5) / 127.5  # Normalize images
        images.append(img)
    with open(dir + '/radiuses.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header
        for row in reader:
            labels.append(int(row[0]))  # Label is the radius of the circle
    return np.array(images), np.array(labels)


# parameters: 772545
def create_cgan_discriminator():
    # Image input
    image_input = Input(shape=(128, 128, 1))
    condition_input = Input(shape=(1,), dtype='float32')  # Change dtype to float32 for radius
    
    # Condition expansion and transformation
    label_embedding = layers.Embedding(input_dim=100, output_dim=50)(condition_input)
    label_embedding = layers.Dense(128 * 128)(label_embedding)
    label_embedding = layers.Reshape((128, 128, 1))(label_embedding)
    
    # Merge inputs
    merged_input = layers.Concatenate()([image_input, label_embedding])
    
    # Discriminator architecture
    x = layers.Conv2D(64, kernel_size=4, strides=2, padding="same")(merged_input)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(128, kernel_size=4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(256, kernel_size=4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)
    
    return Model([image_input, condition_input], x)


def create_cgan_generator(latent_dim):
    # Inputs
    latent_input = Input(shape=(latent_dim,))
    condition_input = Input(shape=(1,), dtype='float32')  # Change dtype to float32 for radius

    # Use an Embedding layer to transform the condition input
    label_embedding = layers.Embedding(input_dim=100, output_dim=latent_dim)(condition_input)
    label_embedding = layers.Flatten()(label_embedding)  # Flatten the embedding output to match the latent dimension

    # Merge inputs
    merged_input = layers.Concatenate()([latent_input, label_embedding])

    # Model architecture
    x = layers.Dense(8 * 8 * 512)(merged_input)
    x = layers.Reshape((8, 8, 512))(x)
    x = layers.Conv2D(256, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(128, kernel_size=5, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(64, kernel_size=5, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(32, kernel_size=5, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(1, kernel_size=5, padding="same", activation="tanh")(x)
    
    # Create model
    return Model([latent_input, condition_input], x)


discriminator = create_cgan_discriminator()
generator = create_cgan_generator(LATENT_DIM)
discriminator.summary()
generator.summary()


# 生成器のアーキテクチャ図を生成
plot_model(generator, to_file='cGAN_practice/output/architecture/generator_model_v3.png', show_shapes=True, show_layer_names=True)

# 識別器のアーキテクチャ図を生成
plot_model(discriminator, to_file='cGAN_practice/output/architecture/discriminator_model_v3.png', show_shapes=True, show_layer_names=True)



def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss


def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)


# GAN
class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [
            self.d_loss_metric,
            self.g_loss_metric,
        ]

    def train_step(self, data):
        real_images, labels = data
        
        batch_size = tf.shape(real_images)[0]
        labels = tf.reshape(labels, (batch_size, 1))  # ラベルの形状を調整

        # 潜在空間からのサンプリング
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        combined_inputs = [random_latent_vectors, labels]

        # 生成器で偽画像を生成
        generated_images = self.generator(combined_inputs, training=True)

        # 補間データの生成
        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0., 1.)
        interpolated_images = alpha * real_images + (1 - alpha) * generated_images

        with tf.GradientTape() as tape:
            # 識別器の損失計算（実画像と生成画像、両方にラベルを適用）
            real_output = self.discriminator([real_images, labels], training=True)
            fake_output = self.discriminator([generated_images, labels], training=True)
            d_loss = discriminator_loss(real_output, fake_output)

            # 勾配罰則の計算
            with tf.GradientTape() as gp_tape:
                gp_tape.watch(interpolated_images)
                interpolated_output = self.discriminator([interpolated_images, labels], training=True)
            grads = gp_tape.gradient(interpolated_output, [interpolated_images])[0]
            norm_grads = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
            gradient_penalty = tf.reduce_mean((norm_grads - 1.0) ** 2)
            d_loss += LAMBDA_GP * gradient_penalty

        # 識別器の勾配更新
        d_grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_weights))

        # 生成器のトレーニング（再度ラベルを結合して偽画像生成）
        with tf.GradientTape() as tape:
            # 潜在ベクトルとラベルを組み合わせた入力から偽画像を生成
            generated_images = self.generator([random_latent_vectors, labels], training=True)
            # 生成された偽画像と実画像のラベルを識別器に入力し、識別器の出力を得る
            fake_output = self.discriminator([generated_images, labels], training=True)
            # 生成器の損失を計算（識別器を騙すための損失）
            g_loss = generator_loss(fake_output)

        # 生成器の勾配更新
        g_grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_weights))

        # メトリクスの更新
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }


class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img_per_label=3, latent_dim=128, radiuses=[20, 30, 40, 50, 60]):
        self.num_img_per_label = num_img_per_label
        self.latent_dim = latent_dim
        self.radiuses = radiuses

    def on_epoch_end(self, epoch, logs=None):
        # 各半径について、num_img_per_label個の画像を生成
        for radius in self.radiuses:
            # 当該半径の繰り返し
            labels = tf.constant([radius] * self.num_img_per_label)
            labels = tf.reshape(labels, (self.num_img_per_label, 1))  # ラベルの形状を調整

            # 潜在空間からのランダムベクトル生成
            random_latent_vectors = tf.random.normal(shape=(self.num_img_per_label, self.latent_dim))

            # 条件付きで偽画像を生成
            generated_images = self.model.generator([random_latent_vectors, labels], training=False)
            generated_images = (generated_images * 127.5) + 127.5  # [-1, 1]から[0, 255]へスケール変換
            generated_images = tf.cast(generated_images, tf.uint8)  # 整数型へ変換

            # 画像を保存
            for i in range(self.num_img_per_label):
                img = keras.preprocessing.image.array_to_img(generated_images[i])
                img.save("cGAN_practice/output/samplev3/generated_img_%03d_radius_%d_%d.png" % (epoch, radius, i))




class LossCSVLogger(keras.callbacks.Callback):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.d_losses = []
        self.g_losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.d_losses.append(logs.get('d_loss'))
        self.g_losses.append(logs.get('g_loss'))

    def on_train_end(self, logs=None):
        with open(self.filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['d_loss', 'g_loss'])
            for d_loss, g_loss in zip(self.d_losses, self.g_losses):
                writer.writerow([d_loss, g_loss])


def plot_latent_space(gan, n=30, figsize=15):
    # display an n*n 2D manifold of digits
    digit_size = 28
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = gan.generator(z_sample)
            digit = tf.reshape(x_decoded[0], (digit_size, digit_size))
            figure[
            i * digit_size : (i + 1) * digit_size,
            j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()


if __name__ == "__main__":
    import os
    import numpy as np


    # 新しいデータセットの読み込み
    dir = 'cGAN_practice/datasets'  # データセットのディレクトリ
    images, labels = load_dataset(dir)
    print("画像の長さ: ", len(images))
    print("ラベルの長さ: ", len(labels))
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(buffer_size=len(images)).batch(BATCH_SIZE)

    output_dir = "cGAN_practice/output/sample"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # train GAN
    gan = GAN(discriminator=discriminator, generator=generator, latent_dim=LATENT_DIM)
    gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    )

    # モデルの重みを読み込む
    # load_dir = "samplev2-batch128-epoch50-lambda10-latent128-data3072"
    # load_dir = "WGAN-gp/output/samplev4"
    # gan.generator.load_weights(os.path.join(load_dir, 'generator_weights.h5'))
    # gan.discriminator.load_weights(os.path.join(load_dir, 'discriminator_weights.h5'))

    gan.fit(
        dataset,
        epochs=EPOCHS,
        callbacks=[
            GANMonitor(latent_dim=LATENT_DIM),
            LossCSVLogger(filename="cGAN_practice/output/samplev3/loss_log.csv")
        ],
    )

    # モデルの重みを保存
    gan.generator.save_weights(os.path.join(output_dir, 'generator_weights.h5'))
    gan.discriminator.save_weights(os.path.join(output_dir, 'discriminator_weights.h5'))

    # 潜在空間のビジュアライズ
    # plot_latent_space(gan)
