'''
demo of a layer with L2 norm
- pred = 0.9, true_label = 1.0
'''
import torch
import torch.nn as nn 

loss_func = nn.BCELoss()

# apply loss
loss = loss_func(torch.tensor([0.9]), torch.tensor([1.0]))

## test for conv layers
conv_layer = nn.Conv2d(in_channels = 3, 
                       out_channels = 5, 
                       kernel_size = 5)

# modify loss with l2 penalty
l2_lambda = 0.001
l2_penalty = l2_lambda \
            * sum( [(p**2).sum() for p in conv_layer.parameters()] )

loss_with_penalty = loss + l2_penalty
print(f'Original Loss: {loss}')
print(f'Loss with L2 penalty (conv layer): {loss_with_penalty}')

