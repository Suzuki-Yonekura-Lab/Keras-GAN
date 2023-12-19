import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# ハイパーパラメータ
BATCH_SIZE = 128
LATENT_DIM = 128  # 潜在空間の次元を適宜変更
EPOCHS = 50

# サンプリングレイヤー
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# エンコーダの作成
encoder_inputs = keras.Input(shape=(32, 32, 3))
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(256, activation="relu")(x)
z_mean = layers.Dense(LATENT_DIM, name="z_mean")(x)
z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

# デコーダの作成
latent_inputs = keras.Input(shape=(LATENT_DIM,))
x = layers.Dense(8 * 8 * 128, activation="relu")(latent_inputs)
x = layers.Reshape((8, 8, 128))(x)
x = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

# VAEモデルの作成
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2, 3)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


def plot_latent_space_cifar10(vae, n=30, figsize=15):
    # display an n*n 2D manifold of CIFAR-10 images
    image_size = 32
    scale = 1.0
    figure = np.zeros((image_size * n, image_size * n, 3))  # 3はRGBチャンネル数

    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample)
            decoded_image = x_decoded[0].reshape(image_size, image_size, 3)
            figure[
            i * image_size : (i + 1) * image_size,
            j * image_size : (j + 1) * image_size,
            :,
            ] = decoded_image

    plt.figure(figsize=(figsize, figsize))
    plt.imshow(figure)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()


def plot_label_clusters_cifar10(vae, data, labels):
    z_mean, _, _ = vae.encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels, cmap='tab10')
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()


if __name__ == "__main__":
    # CIFAR-10データセットの読み込みと前処理
    (x_train, _), (x_test, _) = keras.datasets.cifar10.load_data()
    cifar10_images = np.concatenate([x_train, x_test], axis=0)
    cifar10_images = cifar10_images.astype("float32") / 255  # 画像を0から1の範囲に正規化
    cifar10_images = cifar10_images.reshape((-1, 32, 32, 3))  # 32x32ピクセルのカラー画像

    # VAEモデルのインスタンス化とトレーニング
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam())
    vae.fit(cifar10_images, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # 潜在空間のビジュアライズ
    plot_latent_space_cifar10(vae)

    # ラベルの潜在空間でのビジュアライズ
    (x_train, y_train), _ = keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255  # 画像を0から1の範囲に正規化
    x_train = x_train.reshape((-1, 32, 32, 3))  # 32x32ピクセルのカラー画像

    plot_label_clusters_cifar10(vae, x_train, y_train)
