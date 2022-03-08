import torch

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0

    model.eval()
    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device)
            targets = targets.to(device)

            preds = model(data).argmax(axis=-1)
            num_correct += (preds == targets).sum()

    print(f"Val Acc: {num_correct/len(loader):3.6f}")
