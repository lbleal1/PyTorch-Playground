'''
Implementation of 1D Conv

note: this is the usual conv (outside deep learning)
as seen in the need to flip w
'''

import numpy as np 

def conv1d(x, w, p=0, s=1):
    w_rot = np.array(w[::-1]) # reversed - [1,2,3] -> [3,2,1]
    x_padded = np.array(x) 
    
    # padding
    if p > 0:
        zero_pad = np.zeros(shape=p)
        x_padded = np.concatenate([
            zero_pad, x_padded, zero_pad
        ])
    
    # implementation of formula for conv1d 
    # given rotated w
    res = []
    for i in range(0, int((len(x_padded) - len(w_rot))) + 1, s):
        res.append(np.sum(x_padded[i:i+w_rot.shape[0]] * w_rot))
    
    return np.array(res)

## Testing
x = [1, 3, 2, 4, 5, 6, 1, 3]
w = [1, 0, 3, 1, 2]

# from scratch
print('Conv1d Implementation:', conv1d(x, w, p=2, s=1))

# using numpy 
print('NumPy Results:', np.convolve(x, w, mode='same'))
