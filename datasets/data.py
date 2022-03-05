import os
from PIL import Image

from torch.utils.data import Dataset

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
        
