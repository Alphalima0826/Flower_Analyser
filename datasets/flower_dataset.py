import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms

class FlowerDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        self.flower_types = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
        self.color_types = ["white", "yellow", "red", "pink"]
        self.flower_type_to_idx = {ftype: idx for idx, ftype in enumerate(self.flower_types)}
        self.color_to_idx = {color: idx for idx, color in enumerate(self.color_types)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = f"{self.img_dir}/{row['image_filename']}"
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        flower_label = self.flower_type_to_idx[row['flower_type']]
        color_label = self.color_to_idx[row['dominant_colors']]
        
        oils = torch.tensor([
            row['Linalool'],
            row['Geraniol'],
            row['Citronellol']
        ], dtype=torch.float32)

        return image, flower_label, color_label, oils


def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])