import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
class NumpyDataset(Dataset):
    def __init__(self, npy_path: str | Path, mmap_mode: str | None = 'r'):
        super().__init__()
        self.npy_path = Path(npy_path)
        if not self.npy_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.npy_path}")
        self.data = np.load(self.npy_path, mmap_mode=mmap_mode)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = self.data[idx]
        return torch.from_numpy(np.array(sample))

