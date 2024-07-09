import glob
import csv
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Reshape
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ハイパーパラメータ
BATCH_SIZE = 16
LATENT_DIM = 100
EPOCHS = 100000
LAMBDA_GP = 10


def load_dataset(image_dir, label_path, image_size=(64, 64)):
    # ラベルの読み込み
    df = pd.read_csv(label_path)
    magnet_area_array = np.array(df['magnet_area'].tolist(), dtype=np.float32)
    motor_torque_array = np.array(df['motor_torque'].tolist(), dtype=np.float32)

    # ラベルの正規化（最大値を255にスケーリング）
    magnet_area_array = 255 * (magnet_area_array / np.max(magnet_area_array))
    motor_torque_array = 255 * (motor_torque_array / np.max(motor_torque_array))

    labels_array = np.vstack((magnet_area_array, motor_torque_array)).T

    images = []
    image_files = sorted(glob.glob(os.path.join(image_dir, '*.png')))

    for filename in image_files:
        img = load_img(filename, target_size=image_size, color_mode='rgb')
        img = img_to_array(img, dtype=np.float32)
        img = (img - 127.5) / 127.5  # Normalize images
        images.append(img)

    return np.array(images), labels_array

def create_motor_discriminator(num_classes=2):
    # Image input
    input = Input(shape=(64, 64, 3 + num_classes))

    x = layers.Conv2D(64, kernel_size=4, strides=2, padding="same")(input)
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

    return Model(input, x)

def create_motor_generator(latent_dim, num_classes=2):
    input = Input(shape=(latent_dim + num_classes,))

    x = layers.Dense(4 * 4 * 512)(input)
    x = layers.Reshape((4, 4, 512))(x)
    x = layers.Conv2D(256, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(128, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(64, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(32, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(3, kernel_size=3, padding="same", activation="tanh")(x)

    return Model(input, x)

discriminator = create_motor_discriminator()
generator = create_motor_generator(LATENT_DIM)
discriminator.summary()
generator.summary()

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
        self.train_step_counter = 0

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
        labels = tf.reshape(labels, (batch_size, 2))  # ラベルの形状を調整

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        generator_input = tf.concat([random_latent_vectors, labels], axis=1)
        labels_images = tf.expand_dims(labels, axis=1)
        labels_images = tf.expand_dims(labels_images, axis=2)
        labels_images = tf.tile(labels_images, [1, 64, 64, 1])
        discriminator_real_input = tf.concat([real_images, labels_images], axis=-1)

        generated_images = self.generator(generator_input, training=True)
        discriminator_fake_input = tf.concat([generated_images, labels_images], axis=-1)

        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0., 1.)
        interpolated_images = alpha * real_images + (1 - alpha) * generated_images
        discriminator_interpolated_input = tf.concat([interpolated_images, labels_images], axis=-1)

        if self.train_step_counter % 2 == 0:
            with tf.GradientTape() as tape:
                real_output = self.discriminator(discriminator_real_input, training=True)
                fake_output = self.discriminator(discriminator_fake_input, training=True)
                d_loss = discriminator_loss(real_output, fake_output)

                with tf.GradientTape() as gp_tape:
                    gp_tape.watch(discriminator_interpolated_input)
                    interpolated_output = self.discriminator(discriminator_interpolated_input, training=True)
                grads = gp_tape.gradient(interpolated_output, [discriminator_interpolated_input])[0]
                if grads is None:
                    raise ValueError("グラデーションの計算に失敗しました。グラデーションはNoneです。")
                norm_grads = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
                gradient_penalty = tf.reduce_mean((norm_grads - 1.0) ** 2)
                d_loss += LAMBDA_GP * gradient_penalty

            d_grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_weights))
            self.d_loss_metric.update_state(d_loss)

        with tf.GradientTape() as tape:
            generated_images = self.generator(generator_input, training=True)
            discriminator_fake_input = tf.concat([generated_images, labels_images], axis=-1)
            fake_output = self.discriminator(discriminator_fake_input, training=True)
            g_loss = generator_loss(fake_output)

        g_grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_weights))
        self.g_loss_metric.update_state(g_loss)

        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }


class GANMonitor(keras.callbacks.Callback):
    def __init__(self, dir_name, num_img_per_label=1, latent_dim=64, num_classes=2):
        self.num_img_per_label = num_img_per_label
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.dir_name = dir_name

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            path = f"{self.dir_name}/%04d"
            if not os.path.exists(path % epoch):
                os.makedirs(path % epoch)

            generator_weights_path = os.path.join(path, f'generator.weights.h5')
            self.model.generator.save_weights(generator_weights_path % epoch)


class LossCSVLogger(keras.callbacks.Callback):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        with open(self.filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if csvfile.tell() == 0:
                writer.writerow(['epoch', 'd_loss', 'g_loss'])

    def on_epoch_end(self, epoch, logs=None):
        d_loss = logs.get('d_loss')
        g_loss = logs.get('g_loss')
        with open(self.filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch, d_loss, g_loss])


if __name__ == "__main__":
    import os
    import numpy as np

    name = 'model0123_64x64'
    label_path = f'motor/datasets/{name}/combined_labels.csv'
    image_dir = f'motor/datasets/{name}'
    images, labels = load_dataset(image_dir, label_path)
    print(f'min magnet_area: {min(labels[0])} max magnet_area: {max(labels[0])}')
    print(f'min motor_torque: {min(labels[1])} max motor_torque: {max(labels[1])}')

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(buffer_size=len(images)).batch(BATCH_SIZE).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    output_dir = f'motor/output/{name}_label2'
    loss_csv = f'motor/output/{name}_label2/loss_log.csv'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    gan = GAN(discriminator=discriminator, generator=generator, latent_dim=LATENT_DIM)
    gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    )

    gan.fit(
        dataset,
        epochs=EPOCHS,
        callbacks=[
            GANMonitor(latent_dim=LATENT_DIM, dir_name=output_dir),
            LossCSVLogger(filename=loss_csv)
        ],
    )
