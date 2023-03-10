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

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.layer1(x)
        x = nn.Sigmoid()(x)
        print(x)
        x = self.layer2(x)
        x = nn.Softmax(dim=1)(x)
        return x


input_size = X_test_norm.shape[1]
hidden_size = 16
output_size = 3
model = Model(input_size, hidden_size, output_size)

# load model
model_path = "iris_classifier_params.pt"
model.load_state_dict(torch.load(model_path))

## test saved model
# pred
pred_test = model(X_test_norm)

# calculate accuracy
correct = (torch.argmax(pred_test, dim=1) == y_test).float()
accuracy = correct.mean()
print(f'Test Acc.: {accuracy:.4f}')
