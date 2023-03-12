import torch 
import torch.nn as  nn

# another way of using nn.Sequential
# when the layers have long configs
model = nn.Sequential()

model.add_module('conv1',
                  nn.Conv2d( in_channels = 1, 
                             out_channels = 32, 
                             kernel_size = 5, 
                             padding = 2)
)
model.add_module('relu1', nn.ReLU())
model.add_module('pool1', nn.MaxPool2d(kernel_size = 2))

model.add_module('conv2',
                  nn.Conv2d( in_channels = 32, 
                             out_channels = 64, 
                             kernel_size = 5, 
                             padding = 2)
)
model.add_module('relu2', nn.ReLU())
model.add_module('pool2', nn.MaxPool2d(kernel_size = 2))


model.add_module('flatten', nn.Flatten())

x = torch.ones((4,1,28,28))
print(model(x).shape)


model.add_module('fc1', nn.Linear(3136, 1024))
model.add_module('relu3', nn.ReLU())
model.add_module('dropout', nn.Dropout(p=0.5))
model.add_module('fc2', nn.Linear(1024, 10))

# don't need explicitly add softmax
# - already included in nn.CrossEntropyLoss()
