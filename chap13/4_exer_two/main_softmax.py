'''
Exer 2: classifying MNIST handwritten digits

Focus:
1. data preparation for image data
2. model training using loops and nn.Sequential
'''

import torch
import torch.nn as nn

import torchvision
from torchvision import transforms

from torch.utils.data import DataLoader

# seed
torch.manual_seed(1)

#######################      DATA    ##########################
image_path = './'
transform = transforms.Compose([
    transforms.ToTensor() 
    # converts the pixel features into a floating type tensor
    # normalizes the pixel from the [0, 255] to [0, 1] range
])

mnist_train_dataset = torchvision.datasets.MNIST(root=image_path, 
                                           train=True, 
                                           transform=transform, 
                                           download=True)
mnist_test_dataset = torchvision.datasets.MNIST(root=image_path, 
                                           train=False, 
                                           transform=transform, 
                                           download=False)
 
batch_size = 64
train_dl = DataLoader(mnist_train_dataset, batch_size, shuffle=True)

#######################      MODEL    ##########################
# model structure
hidden_units = [32, 16]
image_size = mnist_train_dataset[0][0].shape
input_size = image_size[0] * image_size[1] * image_size[2]

all_layers = [nn.Flatten()]
for hidden_unit in hidden_units:
    layer = nn.Linear(input_size, hidden_unit)
    all_layers.append(layer)
    all_layers.append(nn.ReLU())
    input_size = hidden_unit

all_layers.append(nn.Linear(hidden_units[-1], 10))
all_layers.append(nn.Softmax())
model = nn.Sequential(*all_layers)
print(model)

# train
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
for epoch in range(num_epochs):
    accuracy_hist_train = 0
    for x_batch, y_batch in train_dl:
        pred = model(x_batch)
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
        accuracy_hist_train += is_correct.sum()
    accuracy_hist_train /= len(train_dl.dataset)
    print(f'Epoch {epoch}  Accuracy {accuracy_hist_train:.4f}')

# test
pred = model(mnist_test_dataset.data / 255.)
is_correct = (torch.argmax(pred, dim=1) == mnist_test_dataset.targets).float()
print(f'Test accuracy: {is_correct.mean():.4f}') 