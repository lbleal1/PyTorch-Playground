from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

import numpy as np

from model import Model

# data 
iris = load_iris()

X = iris['data']
y = iris['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1./3, random_state = 1)

## preprocess

# standardize then convert to tensor
X_test_norm = (X_test - np.mean(X_train))/ np.std(X_train)
X_test_norm = torch.from_numpy(X_test_norm).float()
y_test = torch.from_numpy(y_test)

# load
path = 'iris_classifier.pt'
model_new = torch.load(path)

# verify model architecture
print(model_new.eval())

## test saved model
# pred
pred_test = model_new(X_test_norm)

# calculate accuracy
correct = (torch.argmax(pred_test, dim=1) == y_test).float()
accuracy = correct.mean()
print(f'Test Acc.: {accuracy:.4f}')