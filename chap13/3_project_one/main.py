import pandas as pd 

# for train/test split
import sklearn
import sklearn.model_selection

# for data bucketing and models 
import torch

# for data processing
import torch.nn.functional import one_hot

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

print(f'The processed MPG dataset has the shape: {df.shape}')
print(f'{df.shape[0]} samples')
print(f'{df.shape[1]} features')

## train/test splits
df_train, df_test = sklearn.model_selection.train_test_split(df, train_size = 0.8, random_state=1)
print(f'train shape: {df_train.shape}')
print(f'test shape: {df_test.shape}')

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
total_origin 