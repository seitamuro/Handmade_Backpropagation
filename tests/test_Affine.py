import unittest
from layers import Affine
import numpy as np

class TestAffineLayer(unittest.TestCase):
    def test_init(self):
        a = Affine(np.random.random((1, 1)), np.random.random(1))

    def test_forward(self):
        W = np.array([[1, 2, 3]])
        b = np.array([1, 2, 3])
        x = np.array([1])
        y = np.array([2, 4, 6])
        layer = Affine(W, b)
        self.assertTrue((y == layer.forward(x)).all())



