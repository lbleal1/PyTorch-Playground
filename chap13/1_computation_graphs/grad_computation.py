import torch 

# compute z = wx + b 
# single loss - (y-z)^2
# overall loss - summ (y-z)^2

## assignment
# for grad computation  model parameters
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.5, requires_grad=True)

x = torch.tensor([1.4])
y = torch.tensor([2.1])

## forward pass
# summation
z = torch.add(torch.mul(w,x), b)

# loss
loss = (y-z).pow(2).sum()

# compute gradient
loss.backward()

print('dL/dw :', w.grad)
print('dL/db :', b.grad)
