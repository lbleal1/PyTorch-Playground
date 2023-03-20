import torch

# for the HTTP Error 403
torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
print(model)