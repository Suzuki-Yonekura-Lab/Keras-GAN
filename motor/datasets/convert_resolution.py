import os
import cv2
import pandas as pd

# 元のディレクトリのリスト
source_dirs = [f'motor/datasets/raw/model{model}/seed{i}' for model in range(4) for i in range(3)]
label_csvs = [f'motor/datasets/raw/model{model}/for_label_data.csv' for model in range(4)]

# 新しいディレクトリ
target_dir = 'motor/datasets/mochu'

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
                # 二値化処理
                _, img_binarized = cv2.threshold(img_resized, 127, 255, cv2.THRESH_BINARY)
                # 新しいファイル名
                new_filename = f'{counter:06}.png'
                new_file_path = os.path.join(target_dir, new_filename)
                # 二値化した画像を保存
                cv2.imwrite(new_file_path, img_binarized)
                # cv2.imwrite(new_file_path, img)
                # カウンターをインクリメント
                counter += 1

# 各CSVファイルを結合
df_list = []
for csv_file in label_csvs:
    df = pd.read_csv(csv_file)
    df_list.append(df)

# データフレームを縦方向に結合
combined_df = pd.concat(df_list, ignore_index=True)

# SEEDが0, 1のものだけ抽出
filtered_df = combined_df[combined_df['SEED'].isin([0, 1, 2])]

# 新しいCSVファイルとして保存
combined_csv_path = os.path.join(target_dir, 'combined_labels.csv')
filtered_df.to_csv(combined_csv_path, index=False)

print(f"Resized images and filtered labels have been saved to {target_dir}")
