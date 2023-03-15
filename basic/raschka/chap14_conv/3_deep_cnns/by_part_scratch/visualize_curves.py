import matplotlib.pyplot as plt 

x_arr = np.arange(num_epochs+1)

fig = plt.figure((12,4))
ax = fig.add_subplot(1,2,1)
ax.plot(x_arr, hist[0], '-o', label = 'Train loss')
ax.plot(x_arr, hist[1], '--<', label = 'Valid loss')
ax.legend(fontsize=15)

ax = fig.add_subplot(1,2,2)
ax.plot(x_arr, hist[2], '-o', label = 'Train accuracy')
ax.plot(x_arr, hist[3], '--<', label = 'Valid accuracy')
ax.legend(fontsize=15)

ax.xlabel('Epoch', size=15)
ax.ylabel('Accuracy', size=15)


