import torch
from model import Model

# for the HTTP Error 403
torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

model = torch.hub.load('lbleal1/pytorch-playground', 
                       'custom_model',
                       trust_repo = True)
print(model)