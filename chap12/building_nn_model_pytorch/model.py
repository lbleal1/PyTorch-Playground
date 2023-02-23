import torch

torch.manual_seed(1)

# weight and bias
weight = torch.randn(1)
weight.requires_grad_()
bias = torch.zeros(1, requires_grad = True)

# model
def model(xb):
    return xb @ weight + bias 

# MSE loss
def loss_fn(input, target):
    return(input-target).pow(2).mean()

# training + SGD
learning_rate = 0.001
num_epochs = 200
log_epochs = 10

for epoch in range(num_epochs):

    for x_batch, y_batch in train_dl:
        pred = model(x_batch)
        loss = loss_fn(pred, y_batch)
        loss.backward()
    
    # SGD - optimizer - for weight and bias update
    with torch.no_grad():
        weight -= weight.grad * learning_rate
        bias -= bias.grad * learning_rate
        weight.grad.zero_()
        bias.grad.zero_()
    
    if epoch % log_epochs == 0:
        print(f'Epoch {epoch}   Loss {loss.item():.4f}')