import numpy as np
import unittest
from layers import ReLU

class TestReLULayer(unittest.TestCase):
    def test_init(self):
        layer = ReLU()

    def test_forward_x_gt_0(self):
        layer = ReLU()
        x = np.array([[1, 2, 3]])
        t = np.array([[1, 2, 3]])
        y = layer.forward(x)
        self.assertTrue((y == t).all())

    def test_forward_x_eq_0(self):
        layer = ReLU()
        x = np.array([[0, 0, 0]])
        t = np.array([[0, 0, 0]])
        y = layer.forward(x)
        self.assertTrue((y == t).all())

    def test_forward_x_lt_0(self):
        layer = ReLU()
        x = np.array([[-1, -2, -3]])
        t = np.array([[0, 0, 0]])
        y = layer.forward(x)
        self.assertTrue((y == t).all())

    def test_backward_x_gt_0(self):
        layer = ReLU()
        x = np.array([[1, 2, 3]])
        dout = np.array([[5, 5, 5]])
        t = np.array([[5, 5, 5]])
        layer.forward(x)
        y = layer.backward(dout)
        self.assertTrue((y == t).all())

    def test_backward_x_eq_0(self):
        layer = ReLU()
        x = np.array([[0, 0, 0]])
        dout = np.array([[5, 5, 5]])
        t = np.array([[0, 0, 0]])
        layer.forward(x)
        y = layer.backward(dout)
        self.assertTrue((y == t).all())

    def test_backward_x_lt_0(self):
        layer = ReLU()
        x = np.array([[-1, -2, -3]])
        dout = np.array([[5, 5, 5]])
        t = np.array([[0, 0, 0]])
        layer.forward(x)
        y = layer.backward(dout)
        self.assertTrue((y == t).all())