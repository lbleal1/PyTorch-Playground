import torch
import torch.nn as nn

import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from torchinfo import summary

import os
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(1)
torch.cuda.manual_seed(1)

# config GPU
if torch.cuda.is_available():
    device = torch.device("cuda:0")

# get data
main_dir= '/kaggle/input/intel-image-classification-modified/data'
img_height, img_width = 150, 150
batch_size = 64

# train
train_dir = os.path.join(main_dir, "seg_train")
train_transform = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor() # convert to tensor float, auto normalize [0,1]
    
])
train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
train_dl = DataLoader(train_ds,
                      batch_size,
                      shuffle=True)

# valid
valid_dir = os.path.join(main_dir, "seg_valid")
valid_transform = transforms.Compose([
    transforms.CenterCrop((150,150)),
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor() # convert to tensor float, auto normalize [0,1]
])
valid_ds = datasets.ImageFolder(valid_dir, transform=valid_transform)
valid_dl = DataLoader(valid_ds,
                      batch_size,
                      shuffle=False)


# test
test_dir = os.path.join(main_dir, "seg_test")
test_transform = transforms.Compose([
    transforms.CenterCrop((150,150)),
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor() # convert to tensor float, auto normalize [0,1]
])
test_ds = datasets.ImageFolder(test_dir, transform=test_transform)
test_dl = DataLoader(valid_ds,
                      batch_size,
                      shuffle=False)

# get pretrained model

weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
model = torchvision.models.efficientnet_b0(weights=weights).to(device)

summary(model, 
        input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape" (batch_size, color_channels, height, width)
        verbose=0,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)

# Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
for param in model.features.parameters():
    param.requires_grad = False
    
# Recreate the classifier layer and seed it to the target device
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True), 
    torch.nn.Linear(in_features=1280, 
                    out_features=6, # same number of output units as our number of classes
                    bias=True)).to(device)

summary(model, 
        input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape" (batch_size, color_channels, height, width)
        verbose=0,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

def train(model, num_epochs, train_dl, valid_dl):
    loss_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    
    accuracy_hist_train = [0] * num_epochs
    accuracy_hist_valid = [0] * num_epochs

    for epoch in range(num_epochs):
        model.train() # for dropout + automatic rescaling
        for x_batch, y_batch in tqdm(train_dl):
            # put data batch to gpu
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # y_batch.size(0) is multiplied for normalization
            # see alternative way in xor_nn_3layers.py
            loss_hist_train[epoch] += loss.item()*y_batch.size(0) 
            
            is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
            accuracy_hist_train[epoch] += is_correct.sum().cpu() # need to port back to cpu for plotting

        loss_hist_train[epoch] /= len(train_dl.dataset)
        accuracy_hist_train[epoch] /= len(train_dl.dataset)

        model.eval() # for dropout + automatic rescaling
        with torch.no_grad(): # for dropout
            for x_batch, y_batch in tqdm(valid_dl):
                # put data batch to gpu
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)

                loss_hist_valid[epoch] += loss.item()*y_batch.size(0) 
            
                is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
                accuracy_hist_valid[epoch] += is_correct.sum().cpu() # need to port back to cpu for plotting
        
            loss_hist_valid[epoch] /= len(valid_dl.dataset)
            accuracy_hist_valid[epoch] /= len(valid_dl.dataset)

        print(f'Epoch {epoch+1} accuracy: {accuracy_hist_train[epoch]:.4f} val_accuracy: {accuracy_hist_valid[epoch]:.4f}')
    
    return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid

num_epochs = 5
hist = train(model, num_epochs, train_dl, valid_dl)

# visualize
x_arr = np.arange(num_epochs) + 1

fig = plt.figure(figsize=(12,4))

# loss
ax = fig.add_subplot(1,2,1)
ax.plot(x_arr, hist[0], '-o', label = 'Train loss')
ax.plot(x_arr, hist[1], '--<', label = 'Valid loss')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Loss', size=15)

# accuracy
ax = fig.add_subplot(1,2,2)
ax.plot(x_arr, hist[2], '-o', label = 'Train accuracy')
ax.plot(x_arr, hist[3], '--<', label = 'Valid accuracy')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Accuracy', size=15)
plt.show()

# test 
test_accuracy = 0
for x_batch, y_batch in test_dl:
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    pred = model(x_batch)
    
    is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
    test_accuracy += is_correct.sum()

print(f'Test accuracy: {test_accuracy/len(test_dl.dataset):.4f}')