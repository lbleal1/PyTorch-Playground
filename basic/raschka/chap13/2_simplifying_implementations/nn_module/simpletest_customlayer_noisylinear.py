import torch
import torch.nn as nn

torch.manual_seed(1)

# define custom layer
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

noisy_layer = NoisyLinear(4,2)
x = torch.zeros((1,4))

# x with added noise
print(noisy_layer(x, training = True))
print(noisy_layer(x, training = True))

# just x
print(noisy_layer(x, training = False))