'''
Exer 1: Predicting the Fuel Efficiency of a Car

Focus
1. data preparation for tabular data
2. model training using loops and nn.Sequential
'''
 
import pandas as pd 

# for train/test split
import sklearn
import sklearn.model_selection

# dataset
from torch.utils.data import DataLoader, TensorDataset

# for data bucketing and models 
import torch
import torch.nn as nn

# for data processing
from torch.nn.functional import one_hot

# seed
torch.manual_seed(1)

#######################      DATA    ##########################
## download data
url = 'http://archive.ics.uci.edu/ml/' \
      'machine-learning-databases/auto-mpg/auto-mpg.data'

# read data
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 
                'Weight', 'Acceleration', 'Model Year', 'Origin']
df = pd.read_csv(url, names = column_names,
                na_values = '?', comment = '\t',
                sep = " ", skipinitialspace=True)

## drop the NA rows
df = df.dropna()
df = df.reset_index(drop=True)

'''
print(f'The processed MPG dataset has the shape: {df.shape}')
print(f'{df.shape[0]} samples')
print(f'{df.shape[1]} features')
'''

## train/test splits
df_train, df_test = sklearn.model_selection.train_test_split(df, train_size = 0.8, random_state=1)
'''
print(f'train shape: {df_train.shape}')
print(f'test shape: {df_test.shape}')
'''

## standardizing the continuous features
train_stats = df_train.describe().transpose()
numeric_column_names = ['Cylinders', 'Displacement', 'Horsepower','Weight', 'Acceleration']

df_train_norm, df_test_norm = df_train.copy(), df_test.copy()
for col_name in numeric_column_names:
    mean = train_stats.loc[col_name, 'mean'] 
    std = train_stats.loc[col_name, 'std']

    df_train_norm.loc[:, col_name] = (df_train_norm.loc[:, col_name] - mean) / std
    df_test_norm.loc[:, col_name] = (df_test_norm.loc[:, col_name] - mean) / std

## 'Model Year' into buckets - making it an ordered categorical feature

# establish boundaries - 3 breakpoints, 4 bucket indices starting with bucket index 0
boundaries = torch.tensor([73, 76, 79]) 

# bucket normalized train data
v = torch.tensor(df_train_norm['Model Year'].values) # convert numpy array to tensor
df_train_norm['Model Year Bucketed'] = torch.bucketize( v, boundaries, right=True)

# bucket normalized test data
v = torch.tensor(df_test_norm['Model Year'].values)
df_test_norm['Model Year Bucketed'] = torch.bucketize( v, boundaries, right=True)

# add to numeric_column_names
numeric_column_names.append('Model Year Bucketed')

## one-hot encoding 'Origin' - unordered categorical feature
total_origin = len(set(df_train_norm['Origin']))
origin_encoded = one_hot(torch.from_numpy(df_train_norm['Origin'].values) % total_origin)
x_train_numeric = torch.tensor(df_train_norm[numeric_column_names].values)
x_train = torch.cat([x_train_numeric, origin_encoded], 1).float()
origin_encoded = one_hot(torch.from_numpy(df_test_norm['Origin'].values) % total_origin)
x_test_numeric = torch.tensor(df_test_norm[numeric_column_names].values)
x_test = torch.cat([x_test_numeric, origin_encoded], 1).float()

# create label
y_train = torch.tensor(df_train_norm['MPG'].values).float()
y_test = torch.tensor(df_test_norm['MPG'].values).float()

# data loader
train_ds = TensorDataset(x_train, y_train)
batch_size = 8
train_dl = DataLoader(train_ds, batch_size, shuffle = True)

#######################      MODEL    ##########################
# Way 3: building layers through loops

# model structure
hidden_units = [8, 4]
input_size = x_train.shape[1]
all_layers = []
for hidden_unit in hidden_units:
    layer = nn.Linear(input_size, hidden_unit)
    all_layers.append(layer)
    all_layers.append(nn.ReLU())
    input_size = hidden_unit
all_layers.append(nn.Linear(hidden_units[-1], 1))
model = nn.Sequential(*all_layers) # needs the * to satisfy type - Module subclass - to be passed under nn.Sequential
print(model)

# hyperparams
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

num_epochs = 200
log_epochs = 20 # log every 20 epochs

# train
for epoch in range(num_epochs):
    loss_hist_train = 0
    for x_batch, y_batch in train_dl:
        pred = model(x_batch)[:,0]
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        loss_hist_train += loss.item()

    if epoch % log_epochs == 0:
        print(f'Epoch {epoch} Loss {loss_hist_train/len(train_dl):.4f}')

# test
'''
with torch.no_grad(): # calling this one is quite unnecessary
    pred = model(x_test.float())[:, 0]
    loss = loss_fn(pred, y_test)
    print(f'Test MSE: {loss.item():.4f}')
    print(f'Test MAE: {nn.L1Loss()(pred, y_test).item():.4f}')
'''

pred = model(x_test.float())[:, 0]
loss = loss_fn(pred, y_test)
print(f'Test MSE: {loss.item():.4f}')
print(f'Test MAE: {nn.L1Loss()(pred, y_test).item():.4f}')
