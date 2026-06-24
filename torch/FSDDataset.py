from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import pandas as pd

class FSDDataset(Dataset):
    def __init__(self, csv_file_path, img_dir, transform=None):
        self.data = pd.read_csv(csv_file_path)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        img_path = os.path_join(self.img_dir, row["image"])
        image = Image.open(img_path).convert("RGB")
        
        steering = float(row["steering"])
        throttle = float(row["throttle"])
        label = torch.tensor([steering, throttle], dtype=torch.float32) 
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    


