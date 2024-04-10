import pandas as pd
import torch
from torch.utils.data import Dataset

class CoordinateDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file, header=None, low_memory=False)
        self.transform = transform

    def __len__(self):
        return len(self.data) // 2

    def __getitem__(self, idx):
        idx *= 2
        input_values = torch.tensor(self.data.iloc[idx, 0:].values, dtype=torch.float32) 
        output_values = torch.tensor(self.data.iloc[idx + 1, 0:].values, dtype=torch.float32)

        if self.transform:
            input_values = self.transform(input_values)
            output_values = self.transform(output_values)

        return input_values, output_values
