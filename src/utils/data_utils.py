from torch.utils.data import Dataset
from torch.utils.data import Sampler
import torch

class CycleDatasetWrapper(Dataset):
    def __init__(self, dataset, max_len=1e10):
        self.dataset = dataset
        self.max_len = max_len

    def __getitem__(self, index):
        return self.dataset[index % len(self.dataset)]

    def __len__(self):
        return int(self.max_len)


class InfiniteSampler(Sampler):
    def __init__(self, dataset, shuffle=True, seed=0, max_len=1e10):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.num_samples = len(dataset)
        self.max_len = max_len

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = list(range(self.num_samples))
            if self.shuffle:
                indices = torch.randperm(self.num_samples, generator=g).tolist()
            self.epoch += 1
            yield from indices

    def __len__(self):
        return int(self.max_len)
