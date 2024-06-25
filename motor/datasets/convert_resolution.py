import os
import cv2
import pandas as pd

# 元のディレクトリのリスト
source_dirs = [f'motor/datasets/raw/model{model}/seed{i}' for model in range(4) for i in range(13)]
label_csvs = [f'motor/datasets/raw/model{model}/for_label_data.csv' for model in range(4)]

# 新しいディレクトリ
target_dir = 'motor/datasets/64x64_model0123'

# 新しいディレクトリが存在しない場合は作成
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 6桁の数字を管理する変数
counter = 0

# 各フォルダのpngファイルを処理
for source_dir in source_dirs:
    if os.path.exists(source_dir):
        for filename in os.listdir(source_dir):
            if filename.endswith('.png'):
                # 画像のパス
                file_path = os.path.join(source_dir, filename)
                # 画像を読み込む
                img = cv2.imread(file_path)
                # 画像をリサイズ
                img_resized = cv2.resize(img, (64, 64))
                # 新しいファイル名
                new_filename = f'{counter:06}.png'
                new_file_path = os.path.join(target_dir, new_filename)
                # リサイズした画像を保存
                cv2.imwrite(new_file_path, img_resized)
                # カウンターをインクリメント
                counter += 1

# 各CSVファイルを結合
df_list = []
for csv_file in label_csvs:
    df = pd.read_csv(csv_file)
    df_list.append(df)

# データフレームを縦方向に結合
combined_df = pd.concat(df_list, ignore_index=True)

# 新しいCSVファイルとして保存
combined_csv_path = os.path.join(target_dir, 'combined_labels.csv')
combined_df.to_csv(combined_csv_path, index=False)

print(f"Resized images and combined labels have been saved to {target_dir}")
