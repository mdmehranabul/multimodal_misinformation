# models/dataset.py

import torch
from torch.utils.data import Dataset

class MultimodalDataset(Dataset):
    def __init__(self, dataframe):
        self.texts = dataframe['text_embedding'].values
        self.images = dataframe['image_embedding'].values
        self.labels = dataframe['label'].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text_tensor = torch.tensor(self.texts[idx], dtype=torch.float32)
        image_tensor = torch.tensor(self.images[idx], dtype=torch.float32)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return text_tensor, image_tensor, label_tensor
