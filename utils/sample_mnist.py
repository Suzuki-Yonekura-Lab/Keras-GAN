import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# MNISTデータの読み込み
(x_train, y_train), (_, _) = mnist.load_data()

# いくつかの手書き数字を表示
num_samples = 5
plt.figure(figsize=(10, 4))

for i in range(num_samples):
    ax = plt.subplot(1, num_samples, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')

plt.tight_layout()
plt.show()
