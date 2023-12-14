import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import gdown
from zipfile import ZipFile


# ハイパーパラメータ
BATCH_SIZE = 32
LATENT_DIM = 128
EPOCHS = 50

# Discriminator 作成
discriminator = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ],
    name="discriminator",
)
discriminator.summary()

# Generator 作成
generator = keras.Sequential(
    [
        keras.Input(shape=(LATENT_DIM,)),
        layers.Dense(7 * 7 * 128),
        layers.Reshape((7, 7, 128)),
        layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(1, kernel_size=4, strides=2, padding="same", activation="sigmoid"),
    ],
    name="generator",
)
generator.summary()


# GAN
class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
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

        # 潜在空間状の点をサンプリング
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # 偽画像作成
        generated_images = self.generator(random_latent_vectors)

        # 偽画像と真画像を一つに
        combined_images = tf.concat([generated_images, real_images], axis=0)

        # 真 -> 0, 偽 -> 1
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )

        # ランダムノイズをラベルに付与
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # 潜在空間状の点をサンプリング
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # 全部真画像のラベルを用意
        misleading_labels = tf.zeros((batch_size, 1))

        # train the generator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # update metrics
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
            img.save("generated_img_%03d_%d.png" % (epoch, i))

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
    import numpy as np
    # データセット用意
    # dataset_dir = "celeba_gan"
    #
    # if not os.path.exists(dataset_dir):
    #     os.makedirs(dataset_dir)
    #
    # url = "https://drive.google.com/uc?id=1O7m1010EJjLE5QxLZiM9Fpjs7Oj6e684"
    # output = "celeba_gan/data.zip"
    #
    # if not os.path.exists(output):
    #     gdown.download(url, output, quiet=True)
    #
    #     with ZipFile("celeba_gan/data.zip", "r") as z:
    #         z.extractall("celeba_gan")
    #
    # dataset = keras.utils.image_dataset_from_directory(
    #     "celeba_gan", label_mode=None, image_size=(64, 64), batch_size=BATCH_SIZE
    # )
    # dataset = dataset.map(lambda x: x / 255.0)
    #
    # for x in dataset:
    #     plt.axis("off")
    #     plt.imshow((x.numpy() * 255).astype("int32")[0])
    #     break

    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    mnist_digits = np.concatenate([x_train, x_test], axis=0)
    mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

    # train GAN
    gan = GAN(discriminator=discriminator, generator=generator, latent_dim=LATENT_DIM)
    gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss_fn=keras.losses.BinaryCrossentropy(),
    )

    gan.fit(
        mnist_digits,
        epochs=EPOCHS,
        callbacks=[GANMonitor(num_img=10, latent_dim=LATENT_DIM)],
    )

    # 潜在空間のビジュアライズ
    # plot_latent_space(gan)
