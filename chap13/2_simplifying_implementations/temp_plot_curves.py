import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn

fig = plt.figure(figsize = (16,4))
ax = fig.add_subplot(1, 2, 1)
plt.plot(history[0], lw = 4)
plt.plot(history[1], lw = 4)
plt.legend(['Train loss', 'Validation loss'], fontsize = 15)
ax.set_xlabel('Epochs', size = 15)

ax = fig.add_subplot(1, 2, 2)
plt.plot(history[2], lw = 4)
plt.plot(history[3], lw = 4)
plt.legend(['Train acc.', 'Validation acc.'], fontsize = 15)
ax.set_xlabel('Epochs', size = 15)