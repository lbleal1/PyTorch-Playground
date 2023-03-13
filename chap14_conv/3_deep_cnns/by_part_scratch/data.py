import torch

import torchvision 
from torchvision import transforms 

from torch.utils.data import Subset, DataLoader

# seed
torch.manual_seed(1)

image_path = './'
transform = transforms.Compose([
    # ToTensor() - converts to float tensor, normalizes pixels to [0,1]
    transforms.ToTensor()
])

# load data to train/valid/test splits
mnist_dataset = torchvision.datasets.MNIST(
    root = image_path,
    train = True,
    transform = transform,
    download = True
)
mnist_valid_dataset = Subset(mnist_dataset, 
                             torch.arange(10000))
mnist_train_dataset = Subset(mnist_dataset, 
                             torch.arange(10000, len(mnist_dataset)))

mnist_test_dataset = torchvision.datasets.MNIST(
                                                    root = image_path,
                                                    train = False,
                                                    transform = transform, 
                                                    download = False
)

# dataloader
batch_size = 64
train_dl = DataLoader(mnist_train_dataset,
                      batch_size,
                      shuffle = True)

valid_dl = DataLoader(mnist_valid_dataset,
                      batch_size,
                      shuffle = False)

test_dl = DataLoader(mnist_test_dataset,
                      shuffle = False)

print(test_dl)
print(type(test_dl))