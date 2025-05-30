import torch
from torchvision import transforms

class CatsDataset(torch.utils.data.Dataset):
    def __init__(self, pt_data_path: str):
        super().__init__()
        self.pt_data_path = pt_data_path
        # Shape: (N, 3, 64, 64)
        self.data = torch.load(pt_data_path)
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx] / 255.0
        data = 2 * data - 1
        data = self.transform(data)
        return idx, data
