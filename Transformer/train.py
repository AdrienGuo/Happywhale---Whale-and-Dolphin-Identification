from matplotlib.pyplot import axis
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.cuda.amp import autocast as autocast

from tqdm import tqdm
from models.transformer import ViT
from datasets.dataset import get_loaders
from utils.utils import (check_accuracy)

TRAIN_IMG_DIR = "images/train_images/"
VAL_IMG_DIR = "images/val_images/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 15
NUM_WORKERS = 1

train_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

def train(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    train_loss = 0
    num_correct = 0
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)
        # targets = targets.type(torch.LongTensor)        # PyTorch won't accept a FloatTensor as categorical target

        # forward
        with torch.cuda.amp.autocast():
            outputs = model(data)
            batchLoss = loss_fn(outputs, targets)           # 突然發現怪東西，他這個 outputs 不用取 argmax，那是要怎麼計算阿...
        train_loss += batchLoss.item()
        preds = outputs.argmax(axis=-1)
        num_correct += (preds == targets).sum().item()
        
        # backward
        optimizer.zero_grad()
        scaler.scale(batchLoss).backward()
        scaler.step(optimizer)
        scaler.update()
    print("Train Acc: {:3.6f}, Loss: {:3.6f}".format(num_correct/(len(loader)*BATCH_SIZE), train_loss/len(loader)))


def main():
    train_loader, val_loader, n_classes = get_loaders(
        train_dir=TRAIN_IMG_DIR,
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    print(val_loader.dataset.__getitem__(0))

    model = ViT(
        in_channels=3,
        patch_size=16,
        emb_size=768,
        img_size=224,
        depth=12,
        n_classes=n_classes,
    ).to(device=DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        
        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to console

if __name__ == "__main__":
    main()