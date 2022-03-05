import os
from PIL import Image

from torch.utils.data import Dataset, DataLoader

# 不要直接拿圖片，先從 csv 裡面拿到 image_name，這樣比較好分成 train, validation
# 然後在 __getitem__ 時，再去 train_images directory 裡面拿圖片
class WhaleDolphinDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.imges = os.listdir(img_dir)
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.imges[index])
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)
        
        return img
        # 還有 label 沒有做

def get_loaders(
    train_dir,
    val_dir,
    train_transform,
    val_transform,
    batch_size,
    num_workers=1,
):
    train_dataset = WhaleDolphinDataset(
        img_dir=train_dir,
        transform=train_transform,
    )
    val_dataset = WhaleDolphinDataset(
        img_dir=val_dir,
        transform=val_transform,
    )
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
