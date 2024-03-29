import glob
import csv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt


# ハイパーパラメータ
BATCH_SIZE = 128
LATENT_DIM = 128
EPOCHS = 50
LAMBDA_GP = 10


def load_dataset(image_dir, image_size=(128, 128)):
    images = []
    for img_path in glob.glob(image_dir + '/*/*.png'):
        img = load_img(img_path, target_size=image_size, color_mode='grayscale')
        img = img_to_array(img)
        img = (img - 127.5) / 127.5  # 画像を-1から1の範囲に正規化
        images.append(img)
    return np.array(images)


# 識別器にさらに畳み込み層を追加して強化
# Total params: 722369
discriminator = keras.Sequential([
    layers.Input(shape=(128, 128, 1)),
    layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
    layers.LeakyReLU(alpha=0.2),
    layers.Dropout(0.2),
    layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),  # フィルタ数を増やす
    layers.LeakyReLU(alpha=0.2),
    layers.Dropout(0.2),
    layers.Conv2D(256, kernel_size=4, strides=2, padding="same"),  # 追加の畳み込み層
    layers.LeakyReLU(alpha=0.2),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(1)
], name="discriminator")
discriminator.summary()

# 生成器の修正
# UpSampling2D後のConv2Dのカーネルサイズを調整
# Total params: 6485121
generator = keras.Sequential(
    [
        layers.Input(shape=(LATENT_DIM,)),
        layers.Dense(8 * 8 * 512),
        layers.Reshape((8, 8, 512)),
        layers.Conv2D(256, kernel_size=3, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.UpSampling2D(),
        layers.Conv2D(128, kernel_size=5, padding="same"),  # カーネルサイズを5に調整
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.UpSampling2D(),
        layers.Conv2D(64, kernel_size=5, padding="same"),  # カーネルサイズを5に調整
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.UpSampling2D(),
        layers.Conv2D(32, kernel_size=5, padding="same"),  # カーネルサイズを5に調整
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.UpSampling2D(),
        layers.Conv2D(1, kernel_size=5, padding="same", activation="tanh"),  # カーネルサイズを5に調整
    ],
    name="generator",
)
generator.summary()

# 生成器のアーキテクチャ図を生成
plot_model(generator, to_file='WGAN-gp/output/architecture/generator_model_v4.png', show_shapes=True, show_layer_names=True)

# 識別器のアーキテクチャ図を生成
plot_model(discriminator, to_file='WGAN-gp/output/architecture/discriminator_model_v4.png', show_shapes=True, show_layer_names=True)



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

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]

        # 潜在空間からのサンプリング
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # 生成器で偽画像を生成
        generated_images = self.generator(random_latent_vectors, training=True)

        # 補間データの生成
        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0., 1.)
        interpolated_images = alpha * real_images + (1 - alpha) * generated_images

        with tf.GradientTape() as tape:
            # 識別器の損失計算
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            d_loss = discriminator_loss(real_output, fake_output)

            # 勾配罰則の計算
            with tf.GradientTape() as gp_tape:
                gp_tape.watch(interpolated_images)
                interpolated_output = self.discriminator(interpolated_images, training=True)
            grads = gp_tape.gradient(interpolated_output, [interpolated_images])[0]
            norm_grads = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
            gradient_penalty = tf.reduce_mean((norm_grads - 1.0) ** 2)
            d_loss += LAMBDA_GP * gradient_penalty

        # 識別器の勾配更新
        d_grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_weights))

        # 生成器のトレーニング
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        misleading_labels = tf.zeros((batch_size, 1))

        with tf.GradientTape() as tape:
            fake_output = self.discriminator(self.generator(random_latent_vectors, training=True), training=True)
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
    def __init__(self, num_img=3, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()
        for i in range(self.num_img):
            img = keras.utils.array_to_img(generated_images[i])
            img.save("WGAN-gp/output/samplev4/generated_img_%03d_%d.png" % (epoch, i))


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
    image_dir = 'WGAN-gp/datasets'  # データセットのディレクトリ
    dataset = load_dataset(image_dir)
    print("Dataset shape:", dataset.shape)

    output_dir = "WGAN-gp/output/samplev4"

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
        batch_size=BATCH_SIZE,
        callbacks=[
            GANMonitor(num_img=10, latent_dim=LATENT_DIM),
            LossCSVLogger(filename="WGAN-gp/output/samplev4/loss_log.csv")
        ],
    )

    # モデルの重みを保存
    gan.generator.save_weights(os.path.join(output_dir, 'generator_weights.h5'))
    gan.discriminator.save_weights(os.path.join(output_dir, 'discriminator_weights.h5'))

    # 潜在空間のビジュアライズ
    # plot_latent_space(gan)
