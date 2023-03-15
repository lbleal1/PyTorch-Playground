'''
Implementation of 2D Conv

note: this is the usual conv (outside deep learning)
as seen in the need to flip/rotate W in both dimensions

remarks:
- implementation from scratch is not optimized
- filter matrix is actually not rotated in most tools like PyTorch
'''

import numpy as np
import scipy.signal # for the convolve2d function

def conv2d(X, W, p=(0,0), s=(1,1)):
    W_rot = np.array(W)[::-1, ::-1] # rotating in two axis
    X_orig = np.array (X)

    ## padding
    # padding comp
    n1 = X_orig.shape[0] + 2*p[0]
    n2 = X_orig.shape[1] + 2*p[1]

    # create the blank output
    X_padded = np.zeros(shape=(n1, n2))
    # copy the original X to the blank output - at the center
    X_padded[p[0]:p[0]+X_orig.shape[0],
             p[1]:p[1]+X_orig.shape[1]] = X_orig

    res = []
    for i in range(0, int((X_padded.shape[0] - 
                           W_rot.shape[0])/s[0])+1, s[0]):
        res.append([])
        for j in range(0, int((X_padded.shape[1] - 
                               W_rot.shape[1])/s[1])+1, s[1]):
            X_sub = X_padded[i:i+W_rot.shape[0],
                             j:j+W_rot.shape[1]]
            res[-1].append(np.sum(X_sub * W_rot))
    return(np.array(res))

X = [[1, 3, 2, 4], [5, 6, 1, 3], [1, 2, 0, 2], [3, 4, 3, 2]]
W = [[1, 0, 3], [1, 2, 1], [0, 1, 1]]

print('Conv2d Implementation:\n',
    conv2d(X, W, p=(1, 1), s=(1, 1)))


print('SciPy Results:\n',
    scipy.signal.convolve2d(X, W, mode='same'))
