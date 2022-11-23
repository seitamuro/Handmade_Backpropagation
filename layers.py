import numpy as np

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.dW = None
        self.db = None

    def forward(self, x):
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        self.dW = np.dot(dout, self.W.T)
        self.db = dout
        return self.dW, self.db