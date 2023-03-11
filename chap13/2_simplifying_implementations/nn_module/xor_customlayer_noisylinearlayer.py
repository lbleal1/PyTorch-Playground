'''
- extends xor_nn_3layers.py
- implements a custom layer via NoisyLinear class
    - shows that some layer functionalities can only be used
      during training and NOT during validation or testing
    - implements the noise in w(x+noise) + b during training
    - another related example of this setup is Dropout
- uses the custom layer for the whole neural network

- uses random data for test
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

# data loader - for minibatch
train_ds = TensorDataset(x_train, y_train)
batch_size = 2
train_dl = DataLoader(train_ds, batch_size, shuffle = True)

valid_ds = TensorDataset(x_valid, y_valid)
batch_size = 2
valid_dl = DataLoader(valid_ds, batch_size, shuffle = False) # do not shuffle

#######################     MODEL   ##########################
class NoisyLinear(nn.Module):
    def __init__(self, input_size, output_size, noise_stddev = 0.1):
        super().__init__()
        # torch.Tensor() is equivalent to torch.empty()
        # for creating empty tensors
        w = torch.Tensor(input_size, output_size) 
        # nn.Parameter is a Tensor that's a module parameter
        # adds the variable to the parameter list automatically
        # can be accessed by the 'parameters' iterator
        # has required_grad = True by default
        self.w = nn.Parameter(w)
        nn.init.xavier_uniform_(self.w)

        b = torch.Tensor(output_size).fill_(0)
        self.b = nn.Parameter(b)
        
        self.noise_stddev = noise_stddev

    def forward(self, x, training=False):
        # NEW - applying some methods only in training
        # but not in validation or testing
        # another ex: Dropout

        # only add noise during training
        if training:
            # generate noise from Gaussian Distribution
            noise = torch.normal(0.0, self.noise_stddev, x.shape)
            # x + noise
            x_new = torch.add(x, noise)

        else:
            x_new = x

        return torch.add(torch.mm(x_new, self.w), self.b)

class MyNoisyModule(nn.Module):
    def __init__(self):
        super().__init__()

        # definining layers as attributes of the class
        self.l1 = NoisyLinear(2, 4, 0.07)
        self.a1 = nn.ReLU()
        
        self.l2 = nn.Linear(4, 4)
        self.a2 = nn.ReLU()

        self.l3 = nn.Linear(4, 1)
        self.a3 = nn.Sigmoid()

    def forward(self, x, training=False):
        # we did not use the for loop
        # with self.module_list because the 
        # parameters aren't uniform
        # we have a layer with (x, training)
        # so we have to spell these out
        x = self.l1(x, training)
        x = self.a1(x)

        x = self.l2(x)
        x = self.a2(x)

        x = self.l3(x)
        x = self.a3(x)

        return x

model  = MyNoisyModule()
print(model)


# hyperparams
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.015)

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
        # train - mini-batch version
        for x_batch, y_batch in train_dl:
            pred = model(x_batch, True)[:,0] # forward pass
            loss = loss_fn(pred, y_batch) # calc loss
            loss.backward() # backprop
            optimizer.step() # update weights
            optimizer.zero_grad() # clean gradients
            
            # store loss and acc
            loss_hist_train[epoch] += loss.item()
            
            is_correct = ( (pred >= 0.5).float() == y_batch).float()
            accuracy_hist_train[epoch] += is_correct.mean()

        # train - normalize
        # total number of mini-batches = n_train / batch_size 
        loss_hist_train[epoch] /= n_train/batch_size
        accuracy_hist_train[epoch] /= n_train/batch_size
        
        # validation - mini-batch version
        for x_batch, y_batch in valid_dl: 
            pred = model(x_batch)[:, 0]
            loss = loss_fn(pred, y_batch)
        
            loss_hist_valid[epoch] += loss.item()
            is_correct = ( (pred>= 0.5).float() == y_batch).float()
            accuracy_hist_valid[epoch] += is_correct.mean()

        # valid - normalize
        # total number of mini-batches = n_train / batch_size 
        loss_hist_valid[epoch] /= n_train/batch_size
        accuracy_hist_valid[epoch] /= n_train/batch_size

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