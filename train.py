import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import optim
import sys
sys.path.append('./datasets')
from datasets.flower_dataset import FlowerDataset, get_transforms
from models.model import FlowerMultiOutputModel
device = torch.device("cpu")

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, flower_labels, color_labels, oils in dataloader:
        images = images.to(device)
        flower_labels = flower_labels.to(device)
        color_labels = color_labels.to(device)
        oils = oils.to(device)

        optimizer.zero_grad()
        flower_logits, color_logits, oil_preds = model(images)

        loss_flower = F.cross_entropy(flower_logits, flower_labels)
        loss_color = F.cross_entropy(color_logits, color_labels)
        loss_oils = F.mse_loss(oil_preds, oils)

        loss = loss_flower + loss_color + loss_oils
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def validate(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, flower_labels, color_labels, oils in dataloader:
            images = images.to(device)
            flower_labels = flower_labels.to(device)
            color_labels = color_labels.to(device)
            oils = oils.to(device)

            flower_logits, color_logits, oil_preds = model(images)
            loss_flower = F.cross_entropy(flower_logits, flower_labels)
            loss_color = F.cross_entropy(color_logits, color_labels)
            loss_oils = F.mse_loss(oil_preds, oils)

            loss = loss_flower + loss_color + loss_oils
            running_loss += loss.item() * images.size(0)

    return running_loss / len(dataloader.dataset)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = FlowerDataset(f"{args.data_dir}/train.csv", f"{args.data_dir}/images", transform=get_transforms(train=True))
    val_dataset = FlowerDataset(f"{args.data_dir}/val.csv", f"{args.data_dir}/images", transform=get_transforms(train=False))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = FlowerMultiOutputModel()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} Val Loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("Saved best model.")

if __name__ == '__main__':
    main()