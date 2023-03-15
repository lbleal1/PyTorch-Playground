import torch
import torch.nn as nn 

import numpy as np
import matplotlib.pyplot as plt

# data
from torch.utils.data import TensorDataset, DataLoader

# seed
torch.manual_seed(1)

# data
X_train = np.arange(10, dtype='float32').reshape((10,1))
y_train = np.array([1.0, 1.3, 3.1, 2.0, 5.0, 6.3, 6.6, 7.4, 8.0, 9.0], dtype='float32')

# view data
'''
plt.plot(X_train, y_train, 'o', markersize=10)
plt.xlabel('x'); plt.ylabel('y')
plt.show()
'''

# standardize features - mean centering and dividing by std dev
X_train_norm = (X_train - np.mean(X_train))/np.std(X_train)

# convert to tensor
X_train_norm = torch.from_numpy(X_train_norm)
y_train = torch.from_numpy(y_train)

# create dataset
train_ds = TensorDataset(X_train_norm, y_train)

# load data
batch_size = 1
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

# model setup
input_size = 1
output_size = 1
model = nn.Linear(input_size, output_size)

# train
num_epochs = 200
log_epochs = 10
loss_fn = nn.MSELoss(reduction='mean')
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for x_batch, y_batch in train_dl:
        # generate preds
        y_pred = model(x_batch)
        # calculate loss
        loss = loss_fn(y_pred, y_batch)
        # compute gradients
        loss.backward()
        # update params using gradients
        optimizer.step()
        # reset the grad to zero
        optimizer.zero_grad()

    if epoch % log_epochs == 0:
        print(f'Epoch {epoch} Loss {loss.item():.4f}')


# test
print('Final Parameters:', model.weight.item(), model.bias.item())
X_test = np.linspace(0, 9, num=100, dtype='float32').reshape(-1,1)
X_test_norm = (X_test - np.mean(X_train))/np.std(X_train)
X_test_norm = torch.from_numpy(X_test_norm)

# pred
y_pred = model(X_test_norm).detach().numpy()
print(y_pred.shape)

# plot pred vs orig
fig = plt.figure(figsize=(13,5))
ax = fig.add_subplot(1,2,1)
plt.plot(X_train_norm, y_train, 'o', markersize = 10)
plt.plot(X_test_norm, y_pred, '--', lw = 3)
plt.legend(['Training examples', 'Linear reg.'], fontsize=15)
ax.set_xlabel('x', size = 15) ; ax.set_ylabel('y', size = 15)
ax.tick_params(axis='both', which='major', labelsize=15)
plt.show()