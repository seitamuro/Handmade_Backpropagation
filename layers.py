import numpy as np

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        self.dW = np.dot(self.x.T, dout)
        self.db = dout
        self.dx = np.dot(dout, self.W.T)
        return self.dx

class ReLU:
    def __init__(self):
        pass

    def forward(self, x):
        self.mask = x <= 0
        x[self.mask] = 0
        return x

    def backward(self, dout):
        dout = dout.copy()
        dout[self.mask] = 0
        return dout