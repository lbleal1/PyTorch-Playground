import torch
import torch.nn

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = torch.empty(2, 3, requires_grad = True)
        nn.init.xavier_normal_(self.w1)
        self.w2 = torch.empty(1,2, requires_grad = True)
        nn.init.xavier_normal_(self.w2)