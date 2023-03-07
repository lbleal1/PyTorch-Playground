# creating  special tensor objects
# used to handle gradient computations--store and update

# important note:
# - only tensors of floating point and complex dtype can require gradients

import torch

# 1. Setting requires_grad = True at the same time of creating the variable
# explicit assignment

# example 1 - scalar
a = torch.tensor(3.14)
print('a is:')
print(a)

b = torch.tensor(3.14, requires_grad = True)
print('b is:')
print(b)

print()

# example 2 - list
a = torch.tensor([1.0, 2.0, 3.0], requires_grad = True)
print('now a is a list:')
print(a)
print()

# 2. Setting requires_grad = True after the creation of the variable
# works similarly to type-casting
w = torch.tensor([1.0, 2.0, 3.0])
print(w.requires_grad) # check if the tensor is enabled for gradient computation

# enable the existing tensor for gradient computation
w.requires_grad_()

# check if the tensor is converted to a special tensor object
print(w.requires_grad)