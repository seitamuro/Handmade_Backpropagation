import numpy as np

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    def forward(self, x):
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        dW = np.dot(dout, self.W.T)
        db = dout
        return dW, db