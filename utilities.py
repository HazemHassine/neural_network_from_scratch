import numpy as np

def sigmoid(s , deriv=False):
    if deriv:
        return s * (1-s)
    return 1/ (1 - np.exp(-s))

def apply(sigmoid, array):
    cp = array.copy()
    for i in range(len(cp)):
        cp[i] = sigmoid(cp[i])
    return cp

array = np.random.randn(5,3)
print(array,end="\n\ntest")
transformed_array = apply(sigmoid, array)
