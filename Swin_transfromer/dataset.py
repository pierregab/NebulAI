import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json
from PIL import Image

class NebulaDataset(Dataset):
    def __init__(self, annotations_file, transform=None):
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = self.annotations[idx]['file_path']
        img = Image.open(img_path).convert("RGB")
        bbox = torch.tensor(self.annotations[idx]['bbox'], dtype=torch.float32)
        label = torch.tensor(1)  # Assuming a single class 'nebula' with label 1

        if self.transform:
            img = self.transform(img)
        
        return img, bbox, label

def get_data_loaders(annotations_file, batch_size=4, num_workers=2, train_split=0.8):
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Create dataset
    dataset = NebulaDataset(annotations_file, transform=transform)

    # Split dataset into training and validation sets
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader

# Usage
annotations_file = 'annotations.json'
train_loader, val_loader = get_data_loaders(annotations_file)
