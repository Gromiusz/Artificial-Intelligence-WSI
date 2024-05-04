# import math
# import torch.nn as nn
# import torch.nn.functional as F

# class MLP(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.l1 = nn.Linear(12,12)
#         self.l2 = nn.Linear(12,12)
#         self.l3 = nn.Linear(12,12)
#         self.l4 = nn.Linear(12,12)
#         self.l5 = nn.Linear(12,12)
#         self.l6 = nn.Linear(12,12)
#         self.l7 = nn.Linear(12,12)
#         self.l8 = nn.Linear(12,10)
#         self.l9 = nn.Linear(10,8)
#         self.l10 = nn.Linear(8,4)
    
#     def forward(self, x):
#         for layer in [self.l1, self.l2, self.l3, self.l4, self.l5, self.l6, self.l7, self.l8, self.l9, self.l10]:
#             x = F.relu(layer(x))
#         return x


import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x