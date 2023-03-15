import torch
import torch.nn as nn 

# set random seed
torch.manual_seed(1) 

# create an empty tensor
w = torch.empty(2,3)

# apply glorot initialization
nn.init.xavier_normal_(w)
print(w)
