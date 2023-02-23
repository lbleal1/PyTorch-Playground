import torchvision, torch
from itertools import islice # view 18 examples
import matplotlib.pyplot as plt

# get dataset
image_path = "./"
mnist_dataset = torchvision.datasets.MNIST( image_path, 'train', download = True)

assert isinstance(mnist_dataset, torch.utils.data.Dataset)

# see what the data look like
example = next(iter(mnist_dataset))
print(example)

# visualize images
fig = plt.figure(figsize=(15, 6))
for i, (image, label) in islice(enumerate(mnist_dataset), 10):
    ax = fig.add_subplot(2, 5, i+1)
    ax.set_xticks([]); ax.set_yticks([]); 
    ax.imshow(image, cmap='gray_r')
    ax.set_title(f'{label}', size=15)
plt.show()