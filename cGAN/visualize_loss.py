import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルを読み込む
df = pd.read_csv('cGAN/output/samplev3-batch128-epoch50-lambda20-latent128-data6144/loss_log.csv')

# データをプロットする
plt.figure(figsize=(10, 5))
plt.plot(df['d_loss'], label='Discriminator Loss')
plt.plot(df['g_loss'], label='Generator Loss')
plt.title('Training Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
