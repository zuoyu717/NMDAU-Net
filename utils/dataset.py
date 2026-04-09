import os
import numpy as np
import torch
from torch.utils.data import Dataset

class BraTSDataset(Dataset):
    def __init__(self, root):
        self.files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith("_data.npy")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])  # (4,96,96,96)
        seg = np.load(self.files[idx].replace("_data.npy", "_seg.npy"))  # (96,96,96)

        # 标签合并：BraTS标准
        seg[seg == 4] = 3  # 把4转成3，方便训练
        return torch.from_numpy(data).float(), torch.from_numpy(seg).long()