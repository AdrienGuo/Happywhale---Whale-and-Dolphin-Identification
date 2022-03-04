import pathlib
import pandas as pd

data_path = "images/"
train_df = pd.read_csv(data_path + "train.csv")
train_df['path'] = data_path + train_df['image']

print(train_df.head(10))


