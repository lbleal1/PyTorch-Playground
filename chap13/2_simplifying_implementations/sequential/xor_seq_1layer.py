'''
Using mini-batch will:
- NOT alter the results (see properties of summation)
- will only help the RAM

* implemented mini-batch for TRAINING ONLY
'''

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np 

from torch.utils.data import DataLoader, TensorDataset

# set seeds
torch.manual_seed(1)
np.random.seed(1)

#######################      DATA    ##########################
# create data
x = np.random.uniform(low=-1, high=1, size=(200, 2))

y = np.ones(len(x))
y[x[:,0] * x[:,1] < 0] = 0 # sleek - assign 0 to indices with True


# manually split
n_train = 100

x_train = torch.tensor(x[:n_train, :], dtype = torch.float32)
y_train = torch.tensor(y[:n_train], dtype = torch.float32)

x_valid = torch.tensor(x[n_train:, :], dtype = torch.float32)
y_valid = torch.tensor(y[n_train:], dtype = torch.float32)

# plot data
fig = plt.figure(figsize = (6, 6))

# sleek - get slice of a list using a list of booleans
# (cont) via a conditional statement
plt.plot(x[y==0, 0], x[y==0, 1], 'o', alpha = 0.75, markersize = 10)
plt.plot(x[y==1, 0], x[y==1, 1], '<', alpha = 0.75, markersize = 10)

# use $$ for math formatting
plt.xlabel(r'$x_1$', size = 15)
plt.ylabel(r'$x_2$', size = 15)
#plt.show()

# data loader
train_ds = TensorDataset(x_train, y_train)
batch_size = 2
train_dl = DataLoader(train_ds, batch_size, shuffle = True)

#######################     MODEL   ##########################
# model 1 - baseline - simple linear regression
model = nn.Sequential(
    nn.Linear(2, 1),
    nn.Sigmoid()
)
print(model)

# hyperparams
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)

#  train
num_epochs = 200

def train(model, num_epochs, train_dl, x_valid, y_valid):
    # initialize for tracking loss and accuracy 
    loss_hist_train = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    
    loss_hist_valid =  [0] * num_epochs
    accuracy_hist_valid = [0] * num_epochs

    # train start
    for epoch in range(num_epochs):
        # batch proc
        for x_batch, y_batch in train_dl:
            pred = model(x_batch)[:,0] # forward pass
            loss = loss_fn(pred, y_batch) # calc loss
            loss.backward() # backprop
            optimizer.step() # update weights
            optimizer.zero_grad() # clean gradients
            
            # store loss and acc
            loss_hist_train[epoch] += loss.item()
            
            is_correct = ( (pred >= 0.5).float() == y_batch).float()
            accuracy_hist_train[epoch] += is_correct.mean()

        # normalize 
        # total number of mini-batches = n_train / batch_size 
        loss_hist_train[epoch] /= n_train/batch_size
        accuracy_hist_train[epoch] /= n_train/batch_size
        
        # validation - use whole dataset
        pred = model(x_valid)[:, 0]
        loss = loss_fn(pred, y_valid)
        
        loss_hist_valid[epoch] = loss.item()
        is_correct = ( (pred>= 0.5).float() == y_valid).float()
        accuracy_hist_valid[epoch] += is_correct.mean()
        
    return loss_hist_train, loss_hist_valid, \
           accuracy_hist_train, accuracy_hist_valid

history = train(model, num_epochs, train_dl, x_valid, y_valid)


#######################     VIZ   ##########################
fig = plt.figure(figsize = (16,4))
ax = fig.add_subplot(1, 2, 1)
plt.plot(history[0], lw = 4)
plt.plot(history[1], lw = 4)
plt.legend(['Train loss', 'Validation loss'], fontsize = 15)
ax.set_xlabel('Epochs', size = 15)

ax = fig.add_subplot(1, 2, 2)
plt.plot(history[2], lw = 4)
plt.plot(history[3], lw = 4)
plt.legend(['Train acc.', 'Validation acc.'], fontsize = 15)
ax.set_xlabel('Epochs', size = 15)
plt.show()
