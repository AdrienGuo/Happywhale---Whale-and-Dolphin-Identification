import torch
from tqdm import tqdm

def check_accuracy(loader, model, device="cuda"):
    loss, num_correct = 0, 0

    model.eval()
    with torch.no_grad():
        for data, targets in tqdm(loader):
            data = data.to(device)
            targets = targets.to(device)

            outputs = model(data)
            preds = outputs.argmax(axis=-1)
            num_correct += (preds == targets).sum().item()

    print("Val Acc: {:3.6f}".format(num_correct/(len(loader)*32)))
