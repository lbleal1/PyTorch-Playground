# example of building a graph using pytorch
# z = 2 x (a - b) + c

import torch

def compute_z(a, b, c):
    r1 = torch.sub(a, b)
    r2 = torch.mul(2, r1)
    r3 = torch.add(r2, c)
    return r3

print('Scalar inputs:', compute_z( torch.tensor(1), torch.tensor(2), torch.tensor(3) )) 