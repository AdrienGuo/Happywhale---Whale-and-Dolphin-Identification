from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.cuda.amp import autocast as autocast

from models.transformer import ViT
from datasets.dataset import get_loaders
from utils.utils import (check_accuracy)

TRAIN_IMG_DIR = "images/train_images/"
VAL_IMG_DIR = "images/val_images/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 2
NUM_WORKERS = 1

train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

def train(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
        
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


def main():
    train_loader, val_loader, n_classes = get_loaders(
        train_dir=TRAIN_IMG_DIR,
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

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