import torch
import torch.nn as nn

torch.manual_seed(1)



class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        l1 = nn.Linear(2, 4)
        a1 = nn.ReLU()
        
        l2 = nn.Linear(4, 4)
        a2 = nn.ReLU()

        l3 = nn.Linear(4, 1)
        a3 = nn.Sigmoid()

        l = [l1, a1, l2, a2, l3, a3]
        self.module_list = nn.ModuleList(l)

    def forward(self, x):
        for f in self.module_list:
            x = f(x)
        return x

model  = MyModule()

# hyperparameters
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.015)

# train
history = train(model, num_epochs, train_dl, x_valid, y_valid)