import os
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from skimage import io
from sklearn.utils import shuffle
from torch import seed
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader

PATH = "images/"


def CSVPreprocessor():
    df = pd.read_csv(Path(PATH, "train.csv"))
    N_CLASSES = len(df['individual_id'].unique())
    print("N_CLASSES: ", N_CLASSES)                     # 15,587 classes
    labelencoder_id = LabelEncoder()
    df["individual_id_label"] = labelencoder_id.fit_transform(df["individual_id"])
    print(df)

    return df, N_CLASSES


# 不要直接拿圖片，先從 csv 裡面拿到 image col，這樣比較好分成 train, validation
# 然後在 __getitem__ 時，再去 train_images directory 裡面拿圖片
class WhaleDolphinDataset(Dataset):
    def __init__(self, img_dir, df, transform=None):
        self.img_dir = img_dir
        # self.imges = os.listdir(img_dir)
        self.df = df
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.df['image'].iloc[index])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        label = self.df['individual_id_label'].iloc[index]
        torch.LongTensor(label)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, label

def get_loaders(
    train_dir,
    train_transform,
    val_transform,
    batch_size,
    num_workers=4,
):
    df, n_classes = CSVPreprocessor()
    train_df, valid_df = train_test_split(df, test_size=0.2, shuffle=False)
    print(valid_df)
    train_dataset = WhaleDolphinDataset(
        img_dir=train_dir,
        df=train_df,
        transform=train_transform,
    )
    val_dataset = WhaleDolphinDataset(
        img_dir=train_dir,
        df=valid_df,
        transform=val_transform,
    )
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, n_classes

if __name__ == "__main__":
    CSVPreprocessor()
