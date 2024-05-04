# import torch
# from torch.utils.data import Dataset

# class BCDataset(Dataset):
#     def __init__(self, data, labels):
#         self.data = data
#         self.labels = labels

#     def __getitem__(self, index):
#         return self.data[index], self.labels[index]
    
#     def __len__(self):
#         return len(self.data)

import torch
from torch.utils.data import Dataset

class BCDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            'data': torch.tensor(self.data[idx], dtype=torch.float),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }
        return sample
